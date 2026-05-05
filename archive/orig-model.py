
import time
import torch
from torch import nn
import torch.nn.functional as F
from typing import Iterable, Optional

from funasr.register import tables
from funasr.models.ctc.ctc import CTC
from funasr.utils.datadir_writer import DatadirWriter
from funasr.models.paraformer.search import Hypothesis
from funasr.train_utils.device_funcs import force_gatherable
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.metrics.compute_acc import compute_accuracy, th_accuracy
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from utils.ctc_alignment import ctc_forced_align

class SinusoidalPositionEncoder(torch.nn.Module):
    """
    位置編碼器，把 token/frame 在序列中的時間位置轉成 sinusoidal encoding，直接加到輸入特徵上。
    在 forward() 裡，它為長度 T 的輸入建立 1..T 的位置，算出 sin/cos 編碼後回傳 x + position_encoding。
    這讓 encoder 知道「第幾幀」而不只看到內容。
    """

    def __int__(self, d_model=80, dropout_rate=0.1):
        pass

    def encode(
        self, 
        positions: torch.Tensor = None, # shape = [batch, time]，記錄時間幀的絕對位置
        depth: int = None, # 特徵的總維度 (對應 input_dim 或 d_model)
        dtype: torch.dtype = torch.float32 # 資料的型態
    ):
        batch_size = positions.size(0) # position 第 0 個維度（也就是第一個維度）的大小
        positions = positions.type(dtype) 
        device = positions.device

        # ==========================================
        # 計算頻率率縮放比例 (Log Timescale Increment)
        # ==========================================
        """
        根據 Transformer 論文，頻率的週期會從 2π 呈幾何級數增長到 10000 * 2π。
        這裡計算出每次維度遞增時，頻率對數應該衰減的步長。
        """
        log_timescale_increment = torch.log(
            torch.tensor([10000], dtype=dtype, device=device) 
            ) / (depth / 2 - 1)
        
        # ==========================================
        # 產生從快到慢的頻率倒數 (Inverse Timescales)
        # ==========================================
        """
        透過指數運算產生等比數列，這相當於給不同維度分配不同的「轉速」。
        就像時鐘一樣：前面的維度轉得快(秒針)，後面的維度轉得慢(時針)。
        這樣能確保即使句子極長，每個時間點的編碼依然是獨一無二的。
        """
        inv_timescales = torch.exp(
            torch.arange(depth / 2, device=device).type(dtype) * (-log_timescale_increment)
        )

        # 調整資料的形狀（維度）
        inv_timescales = torch.reshape(inv_timescales, [batch_size, -1])
        
        # ==========================================
        # 時間與頻率的廣播碰撞 (Scaled Time)
        # ==========================================
        """
        positions：代表這是「第幾幀」（例如第 1 幀、第 2 幀）的時間位置。 -> positions 形狀: [1, time, 1] 
        inv_timescales：是我們剛剛算出來的，各種不同快慢的頻率（就像秒針、分針等）。 -> inv_timescales 形狀: [1, 1, depth/2]
        透過矩陣相乘與 Broadcasting 機制，瞬間算出每個時間點在各種頻率下的「角度」。
        輸出的 scaled_time 形狀為 [1, time, depth/2]。
        """
        scaled_time = torch.reshape(positions, [1, -1, 1]) * torch.reshape(
            inv_timescales, [1, 1, -1]
        )

        # ==========================================
        # 正弦與餘弦特徵拼接 (Sinusoidal Encoding)
        # ==========================================
        """
        將剛剛算出的角度分別丟進 正弦（torch.sin）和 餘弦（torch.cos）函數，然後在最後一個維度 (dim=2) 拼接（torch.cat）起來。
        這是Transformer 經典的位置編碼做法：一半的特徵維度用 Sine 轉換，一半用 Cosine 轉換，藉此產生出最終那一組獨一無二的數值。
        (depth/2) + (depth/2) = 完美還原出原本的 depth 特徵總維度。
        這種數學設計不僅賦予了絕對位置資訊，還能讓 Attention 機制容易學習「相對距離」。
        這組數值最後就會加到聲音特徵上，讓模型知道現在是第幾幀。
        """
        encoding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
        
        return encoding.type(dtype) # 返回精度幾 default:F32

    def forward(self, x): # x=embedding（hidden representation）: [B, T, D]
        """
        為輸入的聲學特徵加入「位置資訊 (Position Encoding)」。
        這能讓後續的 Self-Attention 機制擁有時間先後的順序概念。
        """
        # 取得特徵的 Batch Size、時間長度 (timesteps) 與特徵維度 (input_dim)
        batch_size, timesteps, input_dim = x.size() 

        """
        建立時間量尺：產生從 1 到 timesteps 的連續整數序列
        [None, :] 等同於 .unsqueeze(0)，用來增加一個 Batch 的維度，使形狀變為 (1, timesteps)。
        例如 timesteps=3 時，會產生張量：[[1, 2, 4]]
        """
        positions = torch.arange(1, timesteps + 1, device=x.device)[None, :]
        
        # 產生正弦/餘弦編碼：將整數位置轉換為具有連續數學性質的 sinusoidal encoding，並確保資料型態 (dtype) 與運算裝置 (device) 跟輸入特徵一致
        position_encoding = self.encode(positions, input_dim, x.dtype).to(x.device)

        # 融合時間與特徵：將「位置編碼」直接與「原始特徵」相加。回傳的特徵不僅包含了聲音內容，還能讓 Encoder 清楚知道這是序列中的「第幾幀」。
        return x + position_encoding


class PositionwiseFeedForward(torch.nn.Module):
    """
    Transformer 標準的逐位置前饋網路 (Position-wise Feed-Forward Network, FFN)
    【核心任務】：
    在 Attention 層讓不同時間幀互相交流資訊後，本層負責對「每一個獨立的時間幀」進行特徵深加工。
    它對每個 time step 獨立做兩層 MLP：Linear -> activation -> dropout -> Linear，
    不混時間軸，只做特徵維度上的非線性變換。通常作用是大幅提升模型的特徵表達能力。
    
    Positionwise feed forward layer.
    Args:
        idim (int): Input dimenstion. -> 輸入特徵的維度大小 (Input dimension)。
        hidden_units (int): The number of hidden units. -> 隱藏層的維度大小 (通常會比 idim 大，用來升維處理複雜特徵)。
        dropout_rate (float): Dropout rate. -> Dropout 機率，用於防止過度擬合。
        activation (callable): 非線性激勵函數，預設使用 ReLU。
    """

    def __init__(self, idim, hidden_units, dropout_rate, activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        # 第一層線性映射：負責「升維」，將特徵投射到高維空間以萃取更複雜的表示
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        # 第二層線性映射：負責「降維」，將高維特徵壓縮回原本的 idim，以便後續進行殘差相加
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        # 隨機丟棄層 (Regularization)
        self.dropout = torch.nn.Dropout(dropout_rate)
        # 非線性激勵函數
        self.activation = activation

    def forward(self, x):
        """Forward function.
        前向傳播函式 順序：
        1. w_1(x): 特徵升維
        2. activation(...): 非線性轉換
        3. dropout(...): 隨機丟棄部分神經元防止死背
        4. w_2(...): 特徵降維並回傳
        """
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class MultiHeadedAttentionSANM(nn.Module):
    """
    這是這份模型最核心的 block，結合了多頭自注意力和 FSMN memory。
    它先用 linear_q_k_v 產生 Q/K/V，再做 scaled dot-product attention；同時把 v 丟進 depthwise Conv1d 做一段局部時序記憶（forward_fsmn），最後把 attention 輸出和 fsmn_memory 相加。
    所以它不是純 Transformer attention，而是「全域關聯 + 局部串流記憶」的混合版，適合 streaming ASR。
    另外它有 forward_chunk()，專門支援 chunk-based 推論與 cache/look-back。
    補充：
    局部時序記憶 (FSMN) 專屬參數
    kernel_size & sanm_shfit：這兩個參數是為了模型內部的「FSMN memory」準備的。
    在實際運算時，模型會把產生出來的 V (Value) 矩陣丟進一個深度卷積（depthwise Conv1d）來獲取局部的時序記憶。
    kernel_size 決定了這個卷積核（也就是局部記憶的感受野）有多大，而 sanm_shfit 則用於控制特徵在時間軸上的偏移。

    Multi-Head Attention layer.
    Args:
        n_head (int): The number of heads. -> 多頭注意力機制的「頭數（Attention Heads）」。這能讓模型同時從不同的特徵子空間去關注語音訊號。
        n_feat (int): The number of features. -> 輸出特徵維度。
        dropout_rate (float): Dropout rate. -> 隨機丟棄率，用來在訓練階段隨機關閉部分神經元，以防止模型過度擬合。

    """

    def __init__(
        self,
        n_head, # 多頭注意力機制的「頭數（Attention Heads）」 
        in_feat, # 輸入特徵維度（Input Feature）
        n_feat, # 輸出特徵維度
        dropout_rate, # 隨機丟棄率
        kernel_size, # 卷積核（也就是局部記憶的感受野）有多大
        sanm_shfit=0, #控制特徵在時間軸上的偏移
        lora_list=None, 
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
    ):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0 # 確認 n_feat(總特徵維度) 可以被 n_head(注意力頭數) 整除
        
        # We assume d_v always equals d_k (我們假設 d_v 始終等於 d_k)
        self.d_k = n_feat // n_head # 計算出每個「頭」負責處理的特徵維度大小 (假設 Q 和 K 的維度相等)
        self.h = n_head 
        
        """
        被註解掉的三個 Linear 與 self.linear_q_k_v：這是一個非常經典且實用的深度學習工程最佳化！傳統教學上，我們通常會為 Query、Key、Value 各自準備一個線性轉換層（就像被註解掉的那三行）。
        但在實作上，為了極大化 GPU 矩陣運算的效率，程式碼選擇使用單一個 self.linear_q_k_v，一口氣把輸入特徵轉換成 3 倍的大小（n_feat * 3），之後在前向傳播時再把它切成 Q、K、V 三份。這能省下很多零碎的運算時間。
        """
        # self.linear_q = nn.Linear(n_feat, n_feat)
        # self.linear_k = nn.Linear(n_feat, n_feat)
        # self.linear_v = nn.Linear(n_feat, n_feat)

        self.linear_out = nn.Linear(n_feat, n_feat) # 將多頭合併後的結果映射回標準輸出維度的線性層。
        self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3) # 把 輸入特徵 轉換成 3 倍的大小
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate) 
        
        """
        FSMN 局部時序記憶模組 (Depthwise Conv1d)：
        groups=n_feat 代表對每個特徵通道獨立進行時間軸卷積。
        在 forward 時，V 矩陣會通過此層以獲取局部時間上下文。

        groups=n_feat：這裡設定了 groups 參數等於特徵維度，這在神經網路中被稱為「深度卷積（Depthwise Convolution）」。
        這意味著卷積核不會跨特徵維度混合資訊，而是針對每一個特徵通道「獨立」在時間軸上滑動看附近的幀，這既能捕捉時間順序關係，運算量又非常小。在後續運算中，產生的 V 矩陣會被送進這裡來獲取局部記憶。
        """
        self.fsmn_block = nn.Conv1d(
            n_feat, n_feat, kernel_size, stride=1, padding=0, groups=n_feat, bias=False
        )

        # Padding 與 時間軸偏移 (Shift) 計算： 確保卷積後的序列長度不變。
        """
        kernel_size 同時一次看多少連續的時間幀
        為了讓輸入跟輸出的時間幀數（長度）保持一致，我們總共需要補上剩下長度的 kernel_size - 1 個零當 Padding
        
        left_padding = (kernel_size - 1) // 2
        物理意義：在預設情況下（沒有偏移），模型希望卷積核能夠「公平地」看見同等數量的過去與未來資訊。
        計算方式：把總共需要的 padding 數量 (kernel_size - 1) 除以 2，並取整數（// 2）。這就是分給左邊（代表過去的時間點）的 padding 數量。
        
        left_padding = left_padding + sanm_shfit
        物理意義：這正是 SenseVoice 為了支援「低延遲 / 串流推論」所做的特殊設計。在串流語音中，我們往往還沒拿到「未來」的聲音，或者希望模型盡量少看未來以降低延遲。
        計算方式：如果設定了偏移量 sanm_shfit（原始碼中有拼寫錯誤，應為 shift），程式就會把這個偏移量加到左邊的 padding 上。
        視覺化效果：左邊補的零變多了，這會把卷積核的「感受野」硬生生往歷史時間推。這代表模型在計算當下特徵時，會依賴更多的「過去歷史記憶」，進而減少對「未來特徵」的依賴。
        
        right_padding = kernel_size - 1 - left_padding
        物理意義：計算右邊（代表未來的時間點）還需要補多少零。
        計算方式：這是一個簡單的減法。把我們一開始說的總 padding 需求 (kernel_size - 1)，減去剛剛算好且可能已經加上偏移量的 left_padding，剩下的就是分配給右邊的 padding。
        結果：如果前面 left_padding 因為偏移而變多了，這裡算出來的 right_padding 就會自然減少。這完美符合了「多看過去、少看未來」的低延遲串流需求。
        """
        left_padding = (kernel_size - 1) // 2 

        if sanm_shfit > 0: # 如果傳入了這個偏移參數，左邊的 padding 就會增加，右邊相對減少。
            left_padding = left_padding + sanm_shfit
        
        right_padding = kernel_size - 1 - left_padding
        
        # 建立 padding 函式，用於卷積前在時間軸左右側補零
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)

    def forward_fsmn(self, inputs, mask, mask_shfit_chunk=None):
        """
        執行 FSMN 局部時序記憶運算。
        在 MultiHeadedAttentionSANM 中，通常是將 Value (V) 矩陣作為 inputs 傳入此函式，
        透過深度卷積 (depthwise Conv1d) 提取時間軸上的局部歷史特徵。
        """
        b, t, d = inputs.size() # 輸入特徵的批次大小（batch）、時間長度（time）與特徵維度（dimension）。
        
        # 初始 Mask 過濾：確保輸入的 padding 區域為 0，避免無效特徵參與運算
        if mask is not None:
            """
            維度對齊：原本 mask 形狀為 (batch, time)，透過 reshape 將其擴充為 (batch, time, 1)。
            這是為了利用廣播機制，讓 mask 能與形狀為 (batch, time, dim) 的 inputs 順利相乘。
            """
            mask = torch.reshape(mask, (b, -1, 1))
            
            """
            處理串流/分塊推論的特殊遮罩：
            若處於 chunk-based 模式並傳入了 mask_shfit_chunk，則將兩者相乘 (取交集)，確保特徵同時符合有效長度與當前分塊視野。
            """
            if mask_shfit_chunk is not None:
                mask = mask * mask_shfit_chunk
            
            """
            清除雜訊 (Zeroing out)：
            將特徵乘上 mask，強制把 padding 或無效視野區塊的特徵值歸零。
            這是極關鍵的保護步驟！因為後續的 Conv1d 卷積會混合相鄰幀的資訊，若不清零，padding 的雜訊就會在卷積滑動時污染到有效的語音特徵。
            """
            inputs = inputs * mask 

        x = inputs.transpose(1, 2) # PyTorch 的 Conv1d 預設接收的輸入形狀是 (batch, channel, time)，但我們現在的形狀是 (batch, time, dimension)。所以這裡必須把維度 1 和維度 2 交換，讓特徵維度變成 channel。
        x = self.pad_fn(x) # 補零：使用初始化時計算好的 pad_fn (根據 kernel_size 與 shift) 進行補零，確保卷積後長度不變
        x = self.fsmn_block(x) # 局部時序記憶提取：通過一維深度卷積
        x = x.transpose(1, 2) # 卷積做完後，把維度翻轉回原本的 (batch, time, dimension)。
        x += inputs # 殘差連接：將卷積提取出的局部記憶，與原始輸入相加
        x = self.dropout(x) # 過一層 Dropout 防止過度擬合
        
        """
        為什麼還要再做一次呢？因為前面的卷積操作會把相鄰的時間幀資訊混合在一起，這可能會導致原本應該是零的 padding 區域「沾染」到有效區域的數值。
        所以在回傳前，再乘一次 mask，把 padding 區域硬性清零，確保輸出乾淨無瑕，避免雜訊洩漏。
        """
        if mask is not None:
            x = x * mask
        
        return x

    def forward_qkv(self, x):
        """
        將輸入特徵轉換為 Query, Key, Value，並重塑為多頭注意力所需的維度

        Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).
        """
        b, t, d = x.size() # 輸入特徵的 batch、time 和 dimension
        q_k_v = self.linear_q_k_v(x) # 用一個維度 3 倍大的 Linear 層，一口氣把輸入特徵同時轉換成 Q、K、V 的混合體。
        q, k, v = torch.split(q_k_v, int(self.h * self.d_k), dim=-1) # 利用 torch.split 最後一個維度（dim=-1，也就是特徵維度）上，平分成三等份，獨立出 q、k 和 v 張量。
        
        """
        torch.reshape(..., (b, t, self.h, self.d_k))：原本的特徵維度是 h * d_k，這裡透過 reshape 把它強制拆開成兩個維度：self.h（頭的數量）和 self.d_k（每個頭負責的維度）。這賦予了模型「多頭」的能力。
        
        .transpose(1, 2)：把維度 1（時間 t）和維度 2（頭 h）互換。 互換後，形狀從 (batch, time, head, d_k) 變成了 (batch, head, time, d_k)
            為什麼要這樣做？ 因為在 PyTorch 中進行批次矩陣相乘（Batch Matrix Multiplication）時，運算是作用在「最後兩個維度」上的。我們需要讓 Q 和 K 在「時間與特徵」上相乘來計算注意力分數，所以必須把 head 這個維度往前提，把它當作像是 batch 一樣的獨立批次來看待。
        """
        q_h = torch.reshape(q, (b, t, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time1, d_k)
        
        k_h = torch.reshape(k, (b, t, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time2, d_k)
        
        v_h = torch.reshape(v, (b, t, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time2, d_k)

        """
        回傳處理好的多頭 q_h, k_h, v_h 以計算全域注意力。
        同時也回傳「未經多頭轉置」的原始 v，因為 FSMN 局部時序記憶模組 (Conv1d) 需要的是 (batch, time, dim) 形狀的輸入。
        """
        return q_h, k_h, v_h, v

    def forward_attention(self, value, scores, mask, mask_att_chunk_encoder=None):
        # 計算注意力上下文向量 (Scaled Dot-Product Attention 的後半段核心運算)
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """

        n_batch = value.size(0) # 批次大小（Batch Size）：代表模型現在同時正在平行處理幾句話（幾筆語音資料）。
        
        # ==========================================
        # 遮罩處理 (Masking) 與負無限大替換
        # ==========================================
        if mask is not None: 
            # 若有分塊推論遮罩，則與基礎遮罩相乘 (取交集)
            if mask_att_chunk_encoder is not None:
                mask = mask * mask_att_chunk_encoder

            # mask.unsqueeze(1)：在第 1 維插入一個新維度 -> 將 mask 擴充維度以符合多頭矩陣 
            # .eq(0)：等於 0 的地方設為 True，其他為 False -> 找出所有 padding (無效) 的位置
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)

            # 將極小值設為負無限大 (-inf)
            min_value = -float(
                "inf"
            )  # float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            
            # 將 scores 中 padding 位置的分數強制填入負無限大。這樣在下一步做 Softmax (e^x) 時，這些無效位置的機率權重就會變成完美的 0。
            scores = scores.masked_fill(mask, min_value)

            # 計算 Softmax 取得注意力機率分佈，並為了防止浮點數誤差，再次強制將 mask 區域歸零
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            # 若無遮罩，直接計算 Softmax
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        # ==========================================
        # 套用注意力與多頭還原
        # ==========================================
        p_attn = self.dropout(attn) # 在注意力權重上套用隨機丟棄，防止網路過度依賴某些特定的時間幀。

        # 核心：將注意力權重與 Value 矩陣相乘 (Attention Weights * V)
        # 模型是為了更新「某一個特定 frame」的特徵，去參考「整段語音中所有 frame」對這個特定 frame 的注意力權重，然後把「所有 frame」的實際特徵依據這些權重做加權總和（Weighted Sum）
        x = torch.matmul(p_attn, value)  # 輸出形狀: (batch, head, time1, d_k)
        
        """
        復原多頭拆解前的形狀：
        將 time 與 head 維度對調回來 (transpose)
        確保記憶體連續 (contiguous) 後，將多頭特徵重新攤平、連接在一起 (view)
        """
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        # 最後通過線性映射層，融合來自各個注意力頭的資訊並輸出
        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, x, mask, mask_shfit_chunk=None, mask_att_chunk_encoder=None):
        """
        計算縮放點積注意力 (Scaled Dot-Product Attention) 加上 FSMN 局部記憶。
        這是 MultiHeadedAttentionSANM 的主控制流程。

        Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        # ==========================================
        # 特徵準備與局部記憶提取
        # ==========================================
        # 取得多頭的 Q, K, V，以及保留給 FSMN 用的原始 V
        q_h, k_h, v_h, v = self.forward_qkv(x)

        # 將原始 V 送入深度卷積 (Conv1d)，提取「局部時序記憶」
        fsmn_memory = self.forward_fsmn(v, mask, mask_shfit_chunk)
        
        # ==========================================
        # 2. 全域注意力分數計算 (Scaled Dot-Product)
        # ==========================================
        # 縮放 Q 矩陣 (除以 sqrt(d_k))，防止內積數值過大導致 Softmax 梯度消失
        q_h = q_h * self.d_k ** (-0.5)

        # 計算初步的注意力分數 (Query 與 Key 的內積)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        
        # ==========================================
        # 遮罩、加權總和與最終融合
        # ==========================================
        # 把剛剛算好的 scores 加上嚴格的 Padding 遮罩過濾、轉成機率分佈（Softmax），然後與 v_h 進行矩陣相乘加權總和。算出來的 att_outs 就是捕捉了整句話上下文的「全域特徵」。
        att_outs = self.forward_attention(v_h, scores, mask, mask_att_chunk_encoder)
        
        # 將「全域關聯特徵 (att_outs)」與「局部串流記憶 (fsmn_memory)」相加回傳
        return att_outs + fsmn_memory

    def forward_chunk(self, x, cache=None, chunk_size=None, look_back=0):
        # 計算支援串流/分塊 (Chunk-based) 的縮放點積注意力與 FSMN 記憶。
        """
        Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        # 將當前輸入轉換為 Q, K, V
        q_h, k_h, v_h, v = self.forward_qkv(x)
        
        # 歷史快取 (Cache) 拼接與更新機制
        if chunk_size is not None and look_back > 0 or look_back == -1:
            if cache is not None:
                # 擷取當前 chunk 中真正「向前推進」的新鮮特徵 (去除 overlap 區域)
                k_h_stride = k_h[:, :, : -(chunk_size[2]), :]
                v_h_stride = v_h[:, :, : -(chunk_size[2]), :]

                # 【核心】：將過去的歷史 K/V 接在當前的 K/V 前面（在時間軸 dim=2 上拼接）。 這樣稍後算 Attention 時，當前的 Q 就能看見過去的上下文，不僅能對照當下的 K 還能對照過去所有的 K！
                k_h = torch.cat((cache["k"], k_h), dim=2)
                v_h = torch.cat((cache["v"], v_h), dim=2)

                # 將當前的新鮮特徵加入 cache 中，準備交給下一棒
                cache["k"] = torch.cat((cache["k"], k_h_stride), dim=2)
                cache["v"] = torch.cat((cache["v"], v_h_stride), dim=2)

                # 記憶體控管：若不是看全部歷史，則裁切掉太舊的記憶，只保留最近 look_back 個 chunk 的範圍
                if look_back != -1:
                    cache["k"] = cache["k"][:, :, -(look_back * chunk_size[1]) :, :]
                    cache["v"] = cache["v"][:, :, -(look_back * chunk_size[1]) :, :]
            else:
                # 若是第一個 chunk，沒有歷史記憶，直接將當前特徵初始化為 cache
                cache_tmp = {
                    "k": k_h[:, :, : -(chunk_size[2]), :],
                    "v": v_h[:, :, : -(chunk_size[2]), :],
                }
                cache = cache_tmp

        # 提取 FSMN 局部時序記憶 (推論階段預設為全有效特徵，不需傳入 mask)
        # 將原始的 v 丟進一維深度卷積去提取局部時序記憶。注意這裡傳入了 None 取代原本的 mask，因為在分塊推論時，傳入的這塊 v 都是有效特徵，不需要額外過濾 Padding。
        fsmn_memory = self.forward_fsmn(v, None)

        """
        數值縮放 (Scaling)
        將 Query 矩陣乘上維度的開根號倒數 (等同於除以 sqrt(d_k))。
        目的：防止特徵維度過大導致內積數值爆炸。若數值過大會使 Softmax 進入平緩區，
        造成梯度消失 (Gradient Vanishing)，使模型無法收斂。這一步確保了訓練穩定性。
        """
        q_h = q_h * self.d_k ** (-0.5)
        
        """
        計算注意力分數 (Dot-Product)
        將縮放後的 Query 與轉置後的 Key 進行矩陣內積相乘。
        目的：計算整段語音中，各個時間幀 (Frame) 之間的「相似度」與「關聯性分數」。
        因為前面的 k_h 已經偷偷接上了歷史記憶，所以這裡算出來的 scores 包含了對過去時間點的注意力分數！
        """
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        
        # 將分數與包含了歷史的 v_h 結合，算出帶有上下文視野的輸出
        att_outs = self.forward_attention(v_h, scores, None)

        # 回傳 Attention 與 FSMN 的融合結果，並將更新後的 cache 遞交給下一回合
        return att_outs + fsmn_memory, cache


class LayerNorm(nn.LayerNorm):
    """
    這是自訂版的 LayerNorm。
    功能跟 PyTorch 的 nn.LayerNorm 一樣，但它在 forward() 裡先把輸入、weight、bias 轉成 float() 再做 normalization，最後再 cast 回原 dtype。
    這通常是為了 mixed precision / 半精度訓練時的數值穩定性。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        # 暫時提升精度 (Upcast)：
        # 強制將 input, weight, bias 轉換為 float32，避免正規化過程中的平方和運算溢位。
        output = F.layer_norm(
            input.float(), # .float() -> 轉換為 FP32
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        
        # 恢復精度 (Downcast)：
        # 運算完成後，將結果轉回輸入原有的資料型態 (例如 FP16)，
        # 以維持後續網路的運算速度與記憶體效率。
        return output.type_as(input) # .type_as() 是 PyTorch 用來「複製資料型態」的便利函式 -> FP16


def sequence_mask(lengths, maxlen=None, dtype=torch.float32, device=None):
    """
    根據每筆輸入長度產生 mask，讓 attention 和後續層知道哪些 frame 是有效的、哪些是 padding。
    """
    # 確認最大長度：若未給定，則取當前 batch 中的最大真實長度作為統一對齊長度
    if maxlen is None:
        maxlen = lengths.max()
    
    # 建立時間刻度：產生如 [0, 1, 2, ..., maxlen-1] 的一維張量
    row_vector = torch.arange(0, maxlen, 1).to(lengths.device)

    # 維度擴展：將一維的 lengths 轉為二維的欄矩陣 (Column Matrix)
    matrix = torch.unsqueeze(lengths, dim=-1)
    
    # 廣播比對魔法 (Broadcasting)：
    # 拿時間刻度去與每個樣本的真實長度比對。只要時間索引「小於」真實長度就是 True (有效)，
    # 大於等於的就是 False (Padding 補零區)。
    """
    PyTorch 在做 < 比較時，自動先把兩邊用 broadcasting 對齊 shape，然後直接做逐元素比較，產生 boolean tensor
    假設：
    row_vector => 原 shape：(5,) -> 擴展後 shape：(1, 5)
    matrix => 原 shape：(3, 1) -> 擴展後 shape：(3, 1)
    broadcast後 -> (3, 5)

    [[0,1,2,3,4] < 3, 
    [0,1,2,3,4] < 5, 
    [0,1,2,3,4] < 2]

    mask = 
    [[T, T, T, F, F],
    [T, T, T, T, T],
    [T, T, F, F, F]]
    """
    mask = row_vector < matrix
    
    # 資源釋放：這只是輔助矩陣，不需要參與反向傳播算梯度，所以從計算圖中分離，避免 mask 被納入反向傳播
    mask = mask.detach()

    # 型態轉換：將布林值 (True/False) 轉換為浮點數 (1.0/0.0) 以便後續進行矩陣相乘與清零操作
    return mask.type(dtype).to(device) if device is not None else mask.type(dtype)


class EncoderLayerSANM(nn.Module):
    """
    單層 encoder block。
    它把一層完整的編碼流程包起來：

    1. self_attn 做 SANM attention
    2. 殘差連接 + dropout + norm
    3. feed_forward 做逐位置 MLP
    4. 再做一次殘差 + norm
        它還支援 stochastic_depth、concat_after，以及 streaming 的 forward_chunk()。
        可以把它視為「一層可重複堆疊的 SANM encoder layer」。
    """
    def __init__(
        self,
        in_size, # 輸入維度
        size, # 這一層的 hidden dimension
        self_attn,
        feed_forward, 
        dropout_rate, # 每個 sub-layer 的 dropout
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0, # Layer-level dropout
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayerSANM, self).__init__()
        self.self_attn = self_attn # 接收並儲存了多頭注意力模組（也就是結合了全域關聯與局部時序記憶的 MultiHeadedAttentionSANM）。
        self.feed_forward = feed_forward # 接收並儲存了前饋神經網路模組（PositionwiseFeedForward），負責在特徵維度上做非線性變換。
        
        # 實例化兩個自訂版的 LayerNorm，分別用於 Attention 與 FFN 的正規化，提升訓練數值穩定性
        self.norm1 = LayerNorm(in_size) # 通常用於注意力機制（Attention）前後的正規化
        self.norm2 = LayerNorm(size) # 用於前饋網路（FFN）前後的正規化 
        
        self.dropout = nn.Dropout(dropout_rate) # 標準的 Dropout 層，用於隨機丟棄神經元以防止過度擬合。
        
        self.in_size = in_size
        self.size = size
        
        self.normalize_before = normalize_before
        self.concat_after = concat_after # 開關，模型會改用「拼接（Concatenation）」的方式來保留更多原始資訊。
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size) # 如果選擇將注意力輸出與原始輸入拼接起來，特徵維度會變成兩倍（size + size）。因此，這裡準備了一個線性映射層（Linear layer），負責把拼接後加倍的維度重新壓縮、轉換回原本的標準 size。
        
        self.stochastic_depth_rate = stochastic_depth_rate # stochastic_depth_rate (隨機深度率)：它允許模型在訓練過程中，以一定的機率直接跳過整個層級模組，藉此讓網路在訓練時變得「較淺」，進而減輕梯度消失與過度擬合的問題。
        self.dropout_rate = dropout_rate

    def forward(self, x, mask, cache=None, mask_shfit_chunk=None, mask_att_chunk_encoder=None):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (batch, time, size) -> 輸入特徵張量，代表「批次大小、時間幀數、特徵維度」.
            mask (torch.Tensor): Mask tensor for the input (#batch, time). -> 輸入的遮罩，標記哪些是有效資料、哪些是 padding，形狀為 (#batch, time)。
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size). -> 支援 chunk-based (串流/分塊) 推論所設計的快取張量，用來帶入前一次計算的記憶狀態，形狀為 (#batch, time - 1, size)。

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """
        
        skip_layer = False # 標記在這次前向傳播中，是否要因為「隨機深度 (Stochastic Depth)」機制而跳過此層。

        """
        stoch_layer_coeff：隨機深度機制的縮放係數。
        在訓練階段，為了防止過度擬合會隨機丟棄某些層。
        如果該層被保留，原本的殘差連接 `x + f(x)` 會被改為 `x + (1 / (1 - p)) * f(x)`，
        這裡的 1.0 是初始值，後續若處於訓練模式，會根據機率 p (stochastic_depth_rate) 進行動態調整，以確保訓練與推論時的數值期望值一致。
        """
        stoch_layer_coeff = 1.0 

        if self.training and self.stochastic_depth_rate > 0: # 訓練模式 and stochastic_depth_rate(丟棄率) > 0 
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate # 隨機產生 0~1 之間的數如果小於 stochastic_depth_rate(丟棄率) 就 True 
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate) # 縮放係數

        if skip_layer: # 如果跳過 且 cache 不是 None 
            if cache is not None: 
                x = torch.cat([cache, x], dim=1) # 把 cache 記憶與當前 x 在時間維度 dim=1 拼接起來，確保資料完整。
            return x, mask # 回傳 輸入特徵x 與 mask，完全不進行 Attention 與 FFN 等運算

        # ==========================================
        # 第一階段：自注意力機制與殘差連接 (Self-Attention & Residual)
        # ==========================================
        residual = x # 原始特徵備份，作為殘差使用
        if self.normalize_before: # 是否開啟了 Pre-LN
            x = self.norm1(x) # 對 x 進行正規化 -> norm1

        if self.concat_after: # 殘差使用拼接
            # x 與 self_attn 最後一個維度(dim=-1)拼接起來
            x_concat = torch.cat(  
                (
                    x,
                    self.self_attn(
                        x,
                        mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    ),
                ),
                dim=-1,
            )
            # 因為 x_concat 維度變為2倍大，使用 concat_linear 重新降維回原本的標準大小。
            # 都乘上 stoch_layer_coeff 是因為彌補「隨機深度（Stochastic Depth）」在訓練時隨機跳過網路層所造成的數值落差。
            if self.in_size == self.size: # 如果 輸入維度 與 這一層的維度 相同
                # 維度相同：將降維後的拼接結果乘上隨機深度係數，再加上殘差
                x = residual + stoch_layer_coeff * self.concat_linear(x_concat) 
            else: # 如果 輸入維度 與 這一層的維度 不同
                # 維度不同：無法相加，直接將降維後的結果輸出
                x = stoch_layer_coeff * self.concat_linear(x_concat)
        
        else: # 標準模式：使用相加來做殘差連接
            if self.in_size == self.size: # 如果 輸入維度 與 這一層的維度 相同
                # 維度相同：Attention 輸出過 Dropout 並乘上係數後，與殘差相加
                x = residual + stoch_layer_coeff * self.dropout(
                    self.self_attn(
                        x,
                        mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    )
                )
            else: # 如果 輸入維度 與 這一層的維度 不同
                # 維度不同：無法相加，直接輸出 Attention 結果
                x = stoch_layer_coeff * self.dropout(
                    self.self_attn(
                        x,
                        mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    )
                )
        
        if not self.normalize_before: # 若未開啟 Pre-LN 要補做正規化 -> norm1
            x = self.norm1(x)

        # ==========================================
        # 第二階段：逐位置前饋神經網路 (Position-wise FFN)
        # ==========================================
        residual = x # 再次備份經過 Attention 處理後的特徵，作為 FFN 階段的殘差
        if self.normalize_before: # 若開啟 Pre-LN 做正規化 -> norm2
            x = self.norm2(x)
        
        """
        1. 通過 feed_forward (針對每個 time step 獨立做特徵維度的非線性變換，提升表達能力)
        2. 經過 dropout 防止過擬合
        3. 乘上隨機深度縮放係數 (stoch_layer_coeff)
        4. 與備份的 residual 相加，完成殘差連接
        """
        x = residual + stoch_layer_coeff * self.dropout(self.feed_forward(x))
        
        if not self.normalize_before: # 若未開啟 Pre-LN 要補做正規化 -> norm2
            x = self.norm2(x)

        # 回傳更新後的高階特徵 x、傳遞用的 mask，以及用於支援 chunk-based (串流/分塊) 推論的 cache 與狀態參數
        return x, mask, cache, mask_shfit_chunk, mask_att_chunk_encoder

    def forward_chunk(self, x, cache=None, chunk_size=None, look_back=0):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """
        """
        計算分塊 (Chunk) 編碼特徵。
        這是專門用於「串流/分塊推論 (Streaming/Chunk-based Inference)」的前向傳播函式。
        
        與標準 forward() 的最大差異：
        1. 移除了 Dropout 與 Stochastic Depth (隨機深度)，因為推論階段不需要這些正則化機制。
        2. 依賴 chunk_size 與 look_back 來控制注意力視野，並透過 cache 傳遞上下文，取代完整的 mask。
        """

        # ==========================================
        # 第一階段：自注意力機制與殘差連接 (Self-Attention & Residual)
        # ==========================================
        residual = x # 原始特徵備份，作為殘差使用
        if self.normalize_before: # 是否開啟了 Pre-LN
            x = self.norm1(x) # 正規化 -> norm1

        if self.in_size == self.size: # 如果 輸入維度 與 這一層的維度 相同
            # 呼叫支援分塊推論的 Attention，傳入前一次的 cache，並取得更新後的 cache
            attn, cache = self.self_attn.forward_chunk(x, cache, chunk_size, look_back)
            x = residual + attn # 維度相同，進行殘差相加

        else: # 如果 輸入維度 與 這一層的維度 不同
            """
            當輸入維度與輸出維度不同時 (例如第一層 encoders0)，無法進行殘差相加。
            直接呼叫 MultiHeadedAttentionSANM 的串流專用推論函式 (forward_chunk)。
            傳入當前的語音區塊 (x) 與過去的記憶 (cache)，並透過 chunk_size 與 look_back 控制注意力視野，
            最終直接輸出運算後的高階特徵 (x) 以及準備遞交給下一塊使用的更新記憶 (cache)。
            """
            x, cache = self.self_attn.forward_chunk(x, cache, chunk_size, look_back)

        if not self.normalize_before: # 若未開啟 Pre-LN 要補做正規化 -> norm1
            x = self.norm1(x)

        # ==========================================
        # 第二階段：逐位置前饋神經網路 (Position-wise FFN)
        # ==========================================
        residual = x # 再次備份經過 Attention 處理後的特徵，作為 FFN 階段的殘差
        if self.normalize_before: # 若開啟 Pre-LN 要做正規化 -> norm2
            x = self.norm2(x)
        
        x = residual + self.feed_forward(x) # 進行 FFN 運算並直接與殘差相加 (推論階段無須 dropout)

        if not self.normalize_before: # 若未開啟 Pre-LN 要補做正規化 -> norm2
            x = self.norm2(x)

        # 回傳處理好的特徵，以及準備遞交給下一個 chunk 使用的更新版 cache
        return x, cache


@tables.register("encoder_classes", "SenseVoiceEncoderSmall") 
class SenseVoiceEncoderSmall(nn.Module):
    """
    整個 encoder 主體。
    它建立多層 EncoderLayerSANM：
    - encoders0：第一層，負責把 input_size 映射到 output_size
    - encoders：後續主幹層
    - tp_encoders：額外的 tail/post-processing layers 在 forward() 裡，它先建立 mask，對輸入乘上 sqrt(output_size)，加位置編碼，再依序跑這幾組 encoder layers，最後做 normalization 並回傳 xs_pad, olens。
        它就是把聲學特徵轉成高階語音表徵的地方。

    Author: Speech Lab of DAMO Academy, Alibaba Group
    SCAMA: Streaming chunk-aware multihead attention for online end-to-end speech recognition
    https://arxiv.org/abs/2006.01713
    """

    def __init__(
        self,
        input_size: int, # 聲音特徵輸入的維度
        output_size: int = 256, # 聲音特徵輸出的維度
        attention_heads: int = 4, # 注意力頭數
        linear_units: int = 2048, # 前饋神經網路（PositionwiseFeedForward）的線性層神經元數量
        num_blocks: int = 6, # 後續的核心主幹層
        tp_blocks: int = 0, # tp_encoders 後處理層的數量
        dropout_rate: float = 0.1,  # 基本的 Dropout（全域/FFN/Linear 常見）-> 隨機把一部分 neuron 設成 0
        positional_dropout_rate: float = 0.1, # 專門作用在「位置編碼（Positional Encoding）」 -> 隨機讓「時間位置資訊」變模糊
        attention_dropout_rate: float = 0.0, # 專門作用在 Attention 權重上 -> 隨機讓一些 attention 連結消失
        stochastic_depth_rate: float = 0.0, # 專門作用在整個 layer（layer-level dropout） -> 隨機「整層」不訓練
        input_layer: Optional[str] = "conv2d", # 輸入怎麼轉成 embedding
        pos_enc_class=SinusoidalPositionEncoder, # 指定使用 SinusoidalPositionEncoder 來產生位置編碼。
        normalize_before: bool = True, # LayerNorm 要放在「子層前面」還是「子層後面」 -> True:(Pre-LN，現在主流)先做 LayerNorm，再進子模組, Fale:(Post-LN，舊版 Transformer)先加 residual，再做 LayerNorm
        concat_after: bool = False, # Self-Attention 的輸出，要怎麼跟原本的輸入（residual）結合 -> Flase:(預設 / 標準 Transformer)直接相加（residual connection）, True:先 concat，再經過 Linear，再加回去
        positionwise_layer_type: str = "linear", # FFN（Feed Forward Network）類型 -> linear：每個 timestep 獨立, conv:可以看前後時間
        positionwise_conv_kernel_size: int = 1, # 只在 conv1d 時有用, kernel size = 1 -> 等於 linear
        padding_idx: int = -1, # padding token 的 index
        kernel_size: int = 11, # 看前後 11 個時間點
        sanm_shfit: int = 0, # SANM 的時間偏移 -> 控制 attention 是不是偏向「過去」或「未來」
        selfattention_layer_type: str = "sanm", # SANM: Self-Attention + convolution memory
        **kwargs,
    ):
        super().__init__()
        self._output_size = output_size # 標準特徵維度 (預設 256)

        self.embed = SinusoidalPositionEncoder() # 正弦位置編碼器 (SinusoidalPositionEncoder)

        self.normalize_before = normalize_before 

        positionwise_layer = PositionwiseFeedForward # 前饋神經網路 (FFN) 的類別為 PositionwiseFeedForward -> 它會在不混合時間軸的情況下，針對每個 frame 獨立進行特徵維度的非線性轉換，以提升模型的表達能力。
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
        )

        encoder_selfattn_layer = MultiHeadedAttentionSANM # 指定自注意力層的類型，這裡使用的是結合 FSMN 的 SANM 架構
        """
        encoder_selfattn_layer_args0：專門給第一層 (encoders0) 使用的參數。
        由於第一層需要負責將原始特徵維度 (input_size) 映射到模型標準維度 (output_size)，
        因此這裡的輸入與輸出維度設定為 (input_size, output_size)。
        """
        encoder_selfattn_layer_args0 = (
            attention_heads,
            input_size,
            output_size,
            attention_dropout_rate,
            kernel_size,
            sanm_shfit,
        )
        """
        encoder_selfattn_layer_args：專門給後續主幹層 (encoders) 與 尾部層 (tp_encoders) 使用的參數。
        因為資料經過第一層後，維度已經統一為 output_size，
        所以後續層的輸入與輸出維度皆維持不變，設定為 (output_size, output_size)。
        """
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            output_size,
            attention_dropout_rate,
            kernel_size,
            sanm_shfit,
        )
        
        # 建立 encoders0 (第一層 Encoder)
        # 主要任務是作為「橋樑」，將聲音原始特徵的 input_size 映射至模型內部標準的 output_size。
        self.encoders0 = nn.ModuleList(
            [
                EncoderLayerSANM(
                    input_size,
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args0),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                )
                for i in range(1)
            ]
        )
        # 建立 encoders (後續主幹層 Encoder)
        # 因為資料經過第一層(encoders0)後維度已經統一為 output_size，這裡的層專注於深度特徵抽取。
        self.encoders = nn.ModuleList(
            [
                EncoderLayerSANM(
                    output_size,
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                )
                for i in range(num_blocks - 1)
            ]
        )

        """
        建立 tp_encoders (尾部/後處理層 Encoder)
        這些層位於主幹層之後，主要負責最後的特徵微調與後處理 (tail/post-processing layers)。
        因為維度已經固定，所以輸入與輸出皆為 output_size，並使用標準的 args。
        注意：預設的 tp_blocks 為 0，代表預設情況下此列表為空，保留了未來加深網路後處理能力的擴展彈性。
        """
        self.tp_encoders = nn.ModuleList(
            [
                EncoderLayerSANM(
                    output_size,
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                )
                for i in range(tp_blocks)
            ]
        )

        # 實例化自訂版的 LayerNorm (針對 output_size 維度)。
        # 自訂版 LayerNorm 會在底層先將數值轉為 float 再計算，主要是為了提升混合精度/半精度訓練時的數值穩定性。
        
        # 負責在主幹層 (encoders) 處理完畢後，對特徵進行最後的正規化。
        self.after_norm = LayerNorm(output_size)
        # 負責在尾部/後處理層 (tp_encoders) 處理完畢後，對特徵進行正規化。 -> 在 forward 階段，經過層層處理與這兩道正規化手續後，編碼器就會輸出最終的高階語音表徵。
        self.tp_norm = LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor, # 已 padding 的輸入序列
        ilens: torch.Tensor, # 每筆資料的「原始長度」（padding 前）
    ):
        """Embed positions in tensor."""
        masks = sequence_mask(ilens, device=ilens.device)[:, None, :] # 建立遮罩 (Mask)

        xs_pad *= self.output_size() ** 0.5 # Transformer 架構中常見的縮放操作。把輸入乘上維度的平方根，是為了穩定後續 Attention 計算時的梯度與數值分佈。

        xs_pad = self.embed(xs_pad) # 把代表序列時間順序的正弦/餘弦編碼加到特徵上，讓模型在處理時知道現在看的是「第幾幀」的聲音，而不只是特徵本身的內容。

        # forward encoder1
        # 將經過位置編碼的初始特徵送入第一層，在這裡完成從原始輸入維度到模型標準 output_size 的映射轉換。
        for layer_idx, encoder_layer in enumerate(self.encoders0):
            encoder_outs = encoder_layer(xs_pad, masks)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]
        # 繼續將特徵送入後續的主幹層進行深度的語音特徵抽取。
        for layer_idx, encoder_layer in enumerate(self.encoders):
            encoder_outs = encoder_layer(xs_pad, masks)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        xs_pad = self.after_norm(xs_pad) # 跑完主幹層後，先進行一次正規化，穩定數值分佈。

        # forward encoder2
        # 在進入尾部處理前，系統會透過加總 mask 的有效標記，重新確認並計算這批資料的有效輸出長度（olens）。
        olens = masks.squeeze(1).sum(1).int()

        # 如果初始化時有設定 tp_blocks（尾部/後處理層），資料會在這裡進行最後的特徵微調與後處理。
        for layer_idx, encoder_layer in enumerate(self.tp_encoders):
            encoder_outs = encoder_layer(xs_pad, masks)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        xs_pad = self.tp_norm(xs_pad) # 完成所有處理後進行最後一次正規化，並將最終的高階語音表徵（xs_pad）以及有效長度（olens）一起回傳。
        return xs_pad, olens


@tables.register("model_classes", "SenseVoiceSmall")
class SenseVoiceSmall(nn.Module):
    """
    最外層完整模型，也是訓練與推論的入口。它負責：
    - 建立 specaug、normalize、encoder、CTC
    - 準備語言、text normalization、event/emo 的 query embedding
    - forward() 時計算兩種 loss
        - loss_ctc：對第 5 個 token 之後的內容做 CTC
        - loss_rich：對前 4 個特殊 token 做 CE loss
    - inference() 時做音訊載入、抽 fbank、encoder、CTC greedy decode，必要時再做 timestamp alignment

    CTC-attention hybrid Encoder-Decoder model
    """
    
    def __init__(
        self,
        # 設定資料擴增（SpecAugment），在訓練時對聲音加上遮罩來讓模型更強健。
        specaug: str = None, 
        specaug_conf: dict = None,  
        # 設定特徵正規化，把聲音數值縮放到合適的範圍。
        normalize: str = None,
        normalize_conf: dict = None,
        # 設定編碼器
        encoder: str = None,
        encoder_conf: dict = None,
        # 設定 CTC 模組的參數，負責處理語音和文字的對齊。
        ctc_conf: dict = None,
        
        input_size: int = 80, # 原始聲音特徵的維度
        vocab_size: int = -1, # 模型總共能辨識的文字和標記數量。 (-1：用來表示「目前尚未決定」或「無效值」)
        ignore_id: int = -1, # 在計算誤差時需要忽略的標記（例如補齊長度用的 padding）。 (-1：確保它絕對不會跟任何真實的字撞號)
        blank_id: int = 0, # CTC 專用的「空白音」標記（預設為 0）， CTC 用來區隔文字的重要設計。
        sos: int = 1, # 代表句子「開始 (Start of Sequence)」的特殊標記。
        eos: int = 2, # 代表句子「結束 (End of Sequence)」的特殊標記。
        length_normalized_loss: bool = False, # 是否要根據句子的長度來平均計算誤差。(只有在訓練階段才會用到)
        **kwargs,
    ):

        super().__init__()
        
        # 參數請對照 Hugginface: FunAudioLLM/SenseVoiceSmall
        #  - config.yaml
        #  - chn_jpn_yue_eng_ko_spectok.bpe.model -> vocab_size:字典大小

        # 負責資料擴增（像是加上聲音遮罩），讓模型更能適應各種不同的干擾狀況
        if specaug is not None:
            specaug_class = tables.specaug_classes.get(specaug)
            specaug = specaug_class(**specaug_conf)
        # (正規化)把聲音數值統整到標準範圍
        if normalize is not None:
            normalize_class = tables.normalize_classes.get(normalize)
            normalize = normalize_class(**normalize_conf)
        
        # 先從註冊表裡找出你指定的編碼器類型
        encoder_class = tables.encoder_classes.get(encoder)
        # 它把我們之前提到的 input_size=80（原始聲音特徵）還有其他設定一起丟進去，正式把這個編碼器給組裝出來。
        encoder = encoder_class(input_size=input_size, **encoder_conf)
        # 處理完聲音後，輸出的特徵維度
        encoder_output_size = encoder.output_size()
        
        # CTC 設定
        if ctc_conf is None:
            ctc_conf = {}
        
        """
        編碼器 (Encoder) → CTC 模組
        vocab_size：設定輸出的維度
        encoder_output_size：設定輸入CTC的維度
        """
        ctc = CTC(odim=vocab_size, encoder_output_size=encoder_output_size, **ctc_conf)
        

        # 參數放到 Class 中
        self.blank_id = blank_id
        self.sos = sos if sos is not None else vocab_size - 1
        self.eos = eos if eos is not None else vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.specaug = specaug
        self.normalize = normalize
        self.encoder = encoder
        self.error_calculator = None

        self.ctc = ctc

        self.length_normalized_loss = length_normalized_loss
        self.encoder_output_size = encoder_output_size

        # 語言種類標籤
        self.lid_dict = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
        # 字典_ID:語言種類標籤 (轉換動作)
        self.lid_int_dict = {24884: 3, 24885: 4, 24888: 7, 24892: 11, 24896: 12, 24992: 13}
        
        # text normalization:控制輸出的文字要不要包含標點符號或進行反向文字正規化（例如將「一百」轉成「100」）。withitn 代表要，woitn 代表不要
        self.textnorm_dict = {"withitn": 14, "woitn": 15}
        # 字典_ID:語言種類標籤 (轉換動作)
        self.textnorm_int_dict = {25016: 14, 25017: 15}

        # 設定「控制標籤專用字典」的總收錄數量與輸出維度
        """
        7 + len(self.lid_dict) + len(self.textnorm_dict)：這是在計算總共需要處理多少個特殊控制標記。它把情緒標籤（剛好有 7 種）、
            語言種類以及文字輸出格式的數量全部加總起來，決定了這個 Embedding 層要準備多少個「座位」。
        input_size：這是規定每個標籤轉換成數學向量後的大小。這樣能確保這些控制指令的維度跟聲音特徵完全一致，方便後續無縫融合在一起計算。
        """
        self.embed = torch.nn.Embedding(7 + len(self.lid_dict) + len(self.textnorm_dict), input_size)
        
        # 情感種類標籤
        self.emo_dict = {"unk": 25009, "happy": 25001, "sad": 25002, "angry": 25003, "neutral": 25004}
        
        # 設定模型訓練時用到的 「標籤平滑損失函數 (Label Smoothing Loss)」
        self.criterion_att = LabelSmoothingLoss(
            size=self.vocab_size, # 告訴函數總共有多少個標記類別需要預測。
            padding_idx=self.ignore_id, # 在計算誤差時需要忽略的標記（例如補齊長度用的 padding）。
            smoothing=kwargs.get("lsm_weight", 0.0), # 它會把正確答案的目標機率稍微分一點點給其他選項，目的是防止模型對自己的預測「過度自信」，能讓模型面對沒看過的資料時表現更好。
            normalize_length=self.length_normalized_loss, # 設定是否要根據輸入的長度來平均計算誤差。
        )
    
    # 一鍵下載與組裝
    @staticmethod
    def from_pretrained(model:str=None, **kwargs):
        from funasr import AutoModel
        model, kwargs = AutoModel.build_model(model=model, trust_remote_code=True, **kwargs)

        return model, kwargs

    # 定義了模型在訓練階段接收資料並計算誤差的流程。
    def forward(
        self,
        speech: torch.Tensor, # 輸入的聲音特徵矩陣（通常是 Fbank 特徵）。
        speech_lengths: torch.Tensor, # 每段聲音的真實長度。這讓模型知道哪裡是真正有聲音的地方，哪裡是為了對齊而補上的無效空白（搭配 length_normalized_loss 來使用）。
        text: torch.Tensor, # 標準答案！包含這段聲音對應的正確標記（包含情緒、語言等特殊標籤與實際文字）。
        text_lengths: torch.Tensor, # 標準答案的真實長度。
        **kwargs,
    ):
        """
        Encoder + Decoder + Calc loss
        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        
        # 資料整理與防呆
        # 檢查長度資料（text_lengths 和 speech_lengths）的形狀。
        # import pdb;
        # pdb.set_trace()
        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]
        
        # 它從輸入的聲音矩陣中，抓出第一個維度的數字，也就是確認目前這個批次（Batch）總共有「幾筆」音檔準備要同時處理。
        batch_size = speech.shape[0]

        # 1. Encoder
        """
        encoder_out：這是 Encoder 聽完聲音並融合了控制標籤後，提煉出來的「高階特徵矩陣」。
        encoder_out_lens：這是經過神經網路處理後，每段聲音特徵對應的「有效長度」。
        """
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths, text)
        
        loss_ctc, cer_ctc = None, None
        loss_rich, acc_rich = None, None
        stats = dict()
        
        # loss_ctc（語音辨識誤差）：接著它會核對第 5 個標籤之後的「純文字」內容，計算聲音與文字有沒有正確對齊。
        loss_ctc, cer_ctc = self._calc_ctc_loss(
            encoder_out[:, 4:, :], encoder_out_lens - 4, text[:, 4:], text_lengths - 4
        )
        
        # loss_rich（特殊功能誤差）：模型會核對答案的前 4 個特殊標籤（如語言、情緒等），用我們剛剛看過的標籤平滑損失函數來算誤差。
        loss_rich, acc_rich = self._calc_rich_ce_loss(
            encoder_out[:, :4, :], text[:, :4]
        )

        # 「純文字聽寫誤差 (loss_ctc)」和「特殊標籤誤差 (loss_rich)」加在一起，合併成一個最終的總誤差 loss
        loss = loss_ctc + loss_rich
        
        # Collect total loss stats
        # detach() 與 clone()：這是一個節省系統資源的技巧。它把各項分數「影印」一份存進成績單，並切斷與底層運算圖的連結。
        #   你在訓練時畫面上看到的進度條、誤差下降圖或準確率 (acc_rich)，就是從這個 stats 抓取出來的數據。
        stats["loss_ctc"] = torch.clone(loss_ctc.detach()) if loss_ctc is not None else None
        stats["loss_rich"] = torch.clone(loss_rich.detach()) if loss_rich is not None else None
        stats["loss"] = torch.clone(loss.detach()) if loss is not None else None
        stats["acc_rich"] = acc_rich

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        """
        force_gatherable：這是一個專門為了多顯卡訓練 (Multi-GPU) 準備的工具。當你在多張顯卡上同時訓練模型時，它能確保每張顯卡算出來的誤差和數據，都能安全且正確地被收集並合併在一起。
        return loss, stats, weight：最後，把總結算誤差、成績單數據以及權重，正式交還給外圍的 FunASR 訓練框架，讓框架接手去更新模型的參數，完成「學習」。
        """
        if self.length_normalized_loss:
            batch_size = int((text_lengths + 1).sum())
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        
        return loss, stats, weight

    # Encode階段
    def encode(
        self,
        speech: torch.Tensor, # 聲音特徵
        speech_lengths: torch.Tensor, # 聲音長度
        text: torch.Tensor, # 標準答案
        **kwargs,
    ):
        """Frontend + Encoder. Note that this method is used by asr_inference.py
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                ind: int
        """

        # Data augmentation
        # specaug (頻譜增強)
        # 只有在訓練階段才會開啟。它會隨機遮蔽掉部分的聲音特徵，藉由這種「刻意干擾」來訓練模型的抗噪與泛化能力，防止它死背訓練資料。
        if self.specaug is not None and self.training:
            speech, speech_lengths = self.specaug(speech, speech_lengths)

        # Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
        # normalize (正規化)：無論是訓練還是推論階段都會執行。它負責將聲音特徵的數值統一縮放到標準的範圍，讓神經網路更容易吸收和計算。
        if self.normalize is not None:
            speech, speech_lengths = self.normalize(speech, speech_lengths)

        """
        torch.rand(1) > 0.2：這是在做訓練時的隨機干擾。模型有 20% 的機率會「故意把真實的語言答案藏起來」，強制把標籤變成 0（也就是我們前面看過字典裡的 auto 自動偵測）。剩下 80% 的情況才會給出正確的語言代號。這樣做是為了逼迫模型不要過度依賴我們給的答案，要真正學會從聲音裡聽出這是什麼語言。
        elf.embed(lids)：決定好要給正確答案還是 auto 之後，就把這個數字丟進我們之前聊過的 Embedding 轉換台，變成能跟聲音特徵融合的數學向量。
        """
        lids = torch.LongTensor(
            [[self.lid_int_dict[int(lid)] if torch.rand(1) > 0.2 and int(lid) in self.lid_int_dict else 0 ] for lid in text[:, 0]]
            ).to(speech.device)
        language_query = self.embed(lids)
        
        """
        text[:, 3] 與字典轉換：從標準答案的第四個位置抽出代表文字格式的標籤（例如指定要不要加標點符號），並透過 textnorm_int_dict 轉換成內部代號。
        self.embed(styles)：丟進轉換台，變成跟聲音特徵維度相同的數學向量。
        """
        styles = torch.LongTensor(
            [[self.textnorm_int_dict[int(style)]] for style in text[:, 3]]
            ).to(speech.device)
        style_query = self.embed(styles)

        """
        torch.cat(...)（最關鍵的一步）：它直接把這個「格式指令向量」拼接到真正的聲音特徵 (speech) 序列中！這代表模型會把這個指令當作一幀「額外的聲音」輸入進去。
        speech_lengths += 1：因為我們偷偷在聲音序列裡多塞了一個指令幀，所以要跟系統報備，把這段聲音的總有效長度加 1。
        """
        speech = torch.cat((style_query, speech), dim=1)
        speech_lengths += 1

        # 模型直接塞入預設的代號 1 和 2（通常代表讓模型「自動偵測」事件與情緒），並將它們轉換成數學向量 event_emo_query。
        event_emo_query = self.embed(
            torch.LongTensor([[1, 2]]).to(speech.device)
            ).repeat(speech.size(0), 1, 1)
        # 「語言指令 (language)」與這兩個「事件/情緒指令」綁在一起，然後直接黏到真正聲音特徵 (speech) 的最前面！
        input_query = torch.cat((language_query, event_emo_query), dim=1)

        # 在聲音開頭硬塞了 3 個指令幀（語言、事件、情緒），所以整段聲音的有效長度必須加 3。
        speech = torch.cat((input_query, speech), dim=1)
        speech_lengths += 3

        # 把這串帶有指令的聲音序列，轉換成語意特徵矩陣 (encoder_out)，並結算最終的有效長度 (encoder_out_lens)。
        encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)

        return encoder_out, encoder_out_lens

    # 純文字聽寫誤差計算(loss_ctc)
    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor, # 高階聲音特徵。
        encoder_out_lens: torch.Tensor, # 高階聲音長度。
        ys_pad: torch.Tensor, # 真正的純文字標準答案
        ys_pad_lens: torch.Tensor, # 真正的純文字標準答案長度
    ):
        # Calc CTC loss
        # 計算聽寫誤差分數 (loss_ctc)
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        """
        驗證/測試階段執行
        argmax：這是做一次最簡單、快速的「猜題」（Greedy Decoding）。它從大腦提煉出的特徵中，直接挑出每個時間點機率最高的那一個字 (ys_hat)。
        error_calculator：最後，把剛剛快速猜出的答案，拿去跟標準答案 (ys_pad) 進行比對，算出 CER（Character Error Rate，字錯率）。
        """
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        
        return loss_ctc, cer_ctc

    # 特殊功能誤差 (loss_rich)
    def _calc_rich_ce_loss(
        self,
        encoder_out: torch.Tensor, # 高階聲音特徵。
        ys_pad: torch.Tensor, # 真正的純文字標準答案
    ):
        # ctc_lo：這是一個簡單的線性轉換層，負責把大腦 (encoder) 提煉出來的特徵，轉換成字典裡每個標籤的「猜測機率」 (decoder_out)。
        decoder_out = self.ctc.ctc_lo(encoder_out)
        
        # 2. Compute attention loss
        # criterion_att：接著用「標籤平滑損失函數 (Label Smoothing Loss)」，把猜測結果跟標準答案 (ys_pad) 進行比對，算出誤差分數 loss_rich。
        loss_rich = self.criterion_att(decoder_out, ys_pad.contiguous())
        
        #th_accuracy：同時計算出模型在猜測這些特殊標籤時的「準確率 (acc_rich)」，這也就是你在訓練進度條上常看到的準確度指標。
        """
        view(-1, self.vocab_size)：把模型給出的預測結果「壓平」成一長串，方便對應字典裡的每一個可能標籤。
        ys_pad.contiguous()：拿出真實的標準答案（也壓平排列），準備跟預測結果進行一對一比對。
        ignore_label=self.ignore_id：告訴準確率計算機「忽略那些用來補齊長度的無意義空白標籤 (Padding)」。如果不忽略這些空白，模型只要猜對一堆空白，準確率就會看起來很高（虛高），這是不客觀的。
        """
        acc_rich = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_pad.contiguous(),
            ignore_label=self.ignore_id,
        )

        return loss_rich, acc_rich

    # 推論
    def inference(
        self,
        data_in, # 輸入的音檔的路徑或是原始的聲音數據。
        data_lengths=None,
        key: list = ["wav_file_tmp_name"],
        tokenizer=None, # 負責在模型用 CTC 猜出數字代碼後，把它們還原成我們看得懂的文字與情緒標籤。
        frontend=None, # 這是聲音前處理工具，負責把原始聲音轉換成模型大腦能看懂的 Fbank 聲學特徵。
        **kwargs,
    ):
        meta_data = {}
        """
        if:你輸入的資料 (data_in) 已經是被處理過的張量，並且標註為 fbank（聲學特徵），模型就會直接跳過抽取聲音特徵的步驟，把資料原封不動收下來當作 speech。
            speech[None, :, :] -> 幫特徵補上模型需要的「批次 (Batch)」維度而已。
        else:
            
        """
        if (
            isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank"
        ):  # fbank
            speech, speech_lengths = data_in, data_lengths
            if len(speech.shape) < 3:
                speech = speech[None, :, :]
            if speech_lengths is None:
                speech_lengths = speech.shape[1]
        else:
            # extract fbank feats
            # 讀取音檔 (load_audio_...)：將原始聲音檔案讀取進記憶體，並處理好聲音的採樣率。
            time1 = time.perf_counter()
            audio_sample_list = load_audio_text_image_video(
                data_in,
                fs=frontend.fs,
                audio_fs=kwargs.get("fs", 16000),
                data_type=kwargs.get("data_type", "sound"),
                tokenizer=tokenizer,
            )
            # 抽取特徵 (extract_fbank)：呼叫我們前面提過的 frontend 工具，把原始聲波真正轉換成模型大腦能看懂的 Fbank 聲學特徵矩陣。
            """
            這個 frontend 是負責執行**「聲音訊號前處理」**的特徵提取模組！
            在 FunASR 框架中，它的核心任務就是把你輸入的原始聲音波形（raw audio waveform），透過數位訊號處理，呼叫 extract_fbank 轉換成神經網路大腦能看懂的 Fbank 聲學特徵矩陣（頻譜圖）
            。包含聲音的採樣率 (fs)、時間幀的位移 (frame_shift) 等底層參數，都是由它來統一控管。
            """
            time2 = time.perf_counter()
            meta_data["load_data"] = f"{time2 - time1:0.3f}"
            speech, speech_lengths = extract_fbank(
                audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend
            )
            # 效能計時 (time.perf_counter)：你會發現它到處在「按碼錶」！模型在這裡精確記錄了「讀取檔案」跟「轉換特徵」分別花了多少時間 (meta_data)，這是為了方便日後分析和優化推論速度。
            time3 = time.perf_counter()
            meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
            meta_data["batch_data_time"] = (
                speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000
            )

        # 推論設定
        speech = speech.to(device=kwargs["device"])
        speech_lengths = speech_lengths.to(device=kwargs["device"])

        language = kwargs.get("language", "auto")
        # language_query:轉換成數學向量，準備黏在聲音特徵的最前面。
        language_query = self.embed(
            torch.LongTensor(
                [[self.lid_dict[language] if language in self.lid_dict else 0]]
            ).to(speech.device)
        ).repeat(speech.size(0), 1, 1)
        
        use_itn = kwargs.get("use_itn", False) # 用來決定最終輸出是否要包含標點符號與數字正規化。
        output_timestamp = kwargs.get("output_timestamp", False) # 決定是否要幫你算出每個字的發音時間。

        textnorm = kwargs.get("text_norm", None)
        if textnorm is None:
            textnorm = "withitn" if use_itn else "woitn"
        # textnorm_query:轉換成數學向量，接黏到剛剛已經加上語言指令的聲音特徵最前面。
        textnorm_query = self.embed(
            torch.LongTensor([[self.textnorm_dict[textnorm]]]).to(speech.device)
        ).repeat(speech.size(0), 1, 1)
        speech = torch.cat((textnorm_query, speech), dim=1)
        speech_lengths += 1

        # 把「事件」和「情緒」設為預設代號 1 和 2，然後跟「語言指令」綁在一起，最後一口氣黏到聲音特徵的最前面，並跟系統報備長度加 3。
        event_emo_query = self.embed(torch.LongTensor([[1, 2]]).to(speech.device)).repeat(
            speech.size(0), 1, 1
        )
        input_query = torch.cat((language_query, event_emo_query), dim=1)
        speech = torch.cat((input_query, speech), dim=1)
        speech_lengths += 3

        # Encoder
        """
        if isinstance(...) 是一個防呆機制。因為有時候大腦（例如在支援串流處理時）會回傳包含額外記憶狀態的多重包裹，這行程式碼能確保我們只抽出最核心的「高階特徵矩陣」(encoder_out)，不被其他雜訊干擾。
        """
        encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        # c. Passed the encoder result and the beam search
        """
        算出所有機率 (log_softmax)：CTC 把大腦提煉好的特徵，轉換成字典裡每一個字（包含情緒、語言等標籤）在每個時間點的「機率分數」。
        ban_emo_unk 的強制規則：這是一個推論時的客製化開關。如果開啟了它，模型就會把「未知情緒 (unk)」這個標籤的機率強制設定為負無限大（-float("inf")，也就是 0% 機率）。
        """
        ctc_logits = self.ctc.log_softmax(encoder_out)
        if kwargs.get("ban_emo_unk", False):
            ctc_logits[:, :, self.emo_dict["unk"]] = -float("inf")

        results = []
        b, n, d = encoder_out.size() # 特徵矩陣大小。b 代表 Batch size（這一批有幾個音檔），n 是時間長度，d 則是特徵的深度（維度）。
        if isinstance(key[0], (list, tuple)):
            key = key[0] # 輸入的音檔名稱
        if len(key) < b:
            key = key * b # 確保 key 是一個單純的列表，並且裡面的檔名數量跟音檔的數量 (b) 一模一樣。如果不夠，就把它複製補齊。
        for i in range(b): 
            # CTC 貪婪解碼 (Greedy Decode)
            x = ctc_logits[i, : encoder_out_lens[i].item(), :] # 取出這個音檔「真正有效」的時間長度，把尾巴用來補齊長度的無意義空白 (Padding) 通通丟掉。
            yseq = x.argmax(dim=-1) # 貪婪解碼，從每個時間點的機率表中，直接挑出分數最高的那一個字典代碼。
            yseq = torch.unique_consecutive(yseq, dim=-1) # CTC 演算法的壓縮步驟，把連續重複的代碼合併起來，例如把模型初步聽寫出的「我我我愛愛你你」合併成「我愛你」。

            # 如果有設定 output_dir 就存進去
            ibest_writer = None
            if kwargs.get("output_dir") is not None:
                if not hasattr(self, "writer"):
                    self.writer = DatadirWriter(kwargs.get("output_dir"))
                ibest_writer = self.writer[f"1best_recog"] # 第一名最佳預測結果 (1best_recog)
            
            """
            mask = yseq != self.blank_id：CTC 演算法在辨識過程中，會安插很多特殊的「空白標籤」作為字與字之間的分隔符號。這行程式碼就像在做篩選，標記出所有「不是空白標籤」的有效位置。
            token_int = yseq[mask].tolist()：接著把這些無效的空白標籤通通丟掉，只保留真正有意義的數字代碼（代表文字、情緒、事件等），最後把它轉換成單純的 Python 列表 (.tolist())。
            """
            mask = yseq != self.blank_id
            token_int = yseq[mask].tolist()

            # Change integer-ids to tokens
            text = tokenizer.decode(token_int) # 剛剛清理乾淨的token_int，正式交給 decode tokenizer，還原成我們真正看得懂的文字串（裡面也會包含預測出的情緒與事件標籤）。
            if ibest_writer is not None:
                ibest_writer["text"][key[i]] = text # 對應著它原本的音檔名稱 (key[i])，整齊地寫進硬碟裡存檔。

            if output_timestamp: # 是否有開啟計算時間戳記
                from itertools import groupby
                timestamp = []
                tokens = tokenizer.text2tokens(text)[4:] # 把翻譯好的整段文字，重新切成一個個的單字陣列。最關鍵的是 [4:]，這是在切除前 4 個特殊控制標籤（語言、文字格式、事件、情緒）。
                
                logits_speech = self.ctc.softmax(encoder_out)[i, 4:encoder_out_lens[i].item(), :] # 算出的原始分數，正式轉換成 0~1 的「真實機率表」，並且精準切掉前 4 個指令標籤與尾巴的無效長度。

                pred = logits_speech.argmax(-1).cpu() # 找出這個乾淨機率表中，每一個時間點最高機率的預測代碼。
                logits_speech[pred==self.blank_id, self.blank_id] = 0 # 如果模型認為某個時間點最高機率是「空白標籤 (Blank)」，它就會強行把這個空白的機率歸零。

                # ctc_forced_align:強制對齊 (Forced Alignment)
                align = ctc_forced_align(
                    logits_speech.unsqueeze(0).float(), # 乾淨的機率表 (logits_speech.unsqueeze(0))：利用 .unsqueeze(0) 補上工具需要的 Batch 維度，並轉成浮點數。
                    torch.Tensor(token_int[4:]).unsqueeze(0).long().to(logits_speech.device), # 純文字代碼 (token_int[4:])：切掉前 4 個指令的純文字代碼，一樣補上 Batch 維度轉成張量。
                    (encoder_out_lens-4).long()[i], # 精確的長度：告訴對齊工具聲音與文字的「真實有效長度」，並且扣除了那 4 個前導指令。
                    torch.tensor(len(token_int)-4).unsqueeze(0).long().to(logits_speech.device), # 精確的長度：告訴對齊工具聲音與文字的「真實有效長度」，並且扣除了那 4 個前導指令。
                    ignore_id=self.ignore_id,
                )

                # 算出的 align 陣列中，同一個字通常會連續佔用好幾個時間幀（例如：[字A, 字A, 字A, 字B, 字B...]）。
                # groupby 會把它們「打包」起來，這樣我們就能輕鬆算出「字A」持續了 3 幀、「字B」持續了 2 幀。
                pred = groupby(align[0, :encoder_out_lens[0]]) 
                _start = 0 # 設定計時與算字的「歸零起跑點」，從第 0 幀
                token_id = 0 # 翻譯出來的第 0 個字開始往後推進。
                ts_max = encoder_out_lens[i] - 4 # 設定這段語音的有效時間上限，並且再次嚴格扣除前面那 4 個控制標籤的時間長度，確保不會算到指令頭上。
                for pred_token, pred_frame in pred:
                    _end = _start + len(list(pred_frame)) # 累加幀數 (_end = _start + len(...))：算出當前這個字總共佔用了幾幀，並標記結束位置。
                    if pred_token != 0 and token_id < len(tokens): # 如果這一段是 CTC 的空白標籤就直接跳過，我們只針對「真正有發音的字」計算時間。
                        # 把幀數乘上 60 再減 30，最後除以 1000。因為聲音經過層層壓縮後，這裡的每一幀大約代表 60 毫秒的時間。這個公式精準地把它轉換成以「秒」為單位的絕對時間。
                        """
                        這個公式確實是 SenseVoiceSmall 架構專屬的！
                        這背後的原理是「特徵壓縮 (Subsampling)」。通常語音模型一開始處理聲音時，會把聲音切成非常細碎的片段（例如每 10 毫秒一幀）。
                        但為了提高運算的效率與速度，聲音特徵在經過 SenseVoiceSmall 的大腦（Encoder）層層提煉時，會被自動壓縮。
                        在這個特定的架構下，它的壓縮比例剛好讓最後輸出的每一幀代表 60 毫秒的時間。
                        所以：
                        乘 60：把幀數還原回實際的毫秒數。
                        減 30：為了抓取這個 60 毫秒區間的「正中間」作為精確的時間標記。
                        除以 1000：把毫秒換算成我們習慣的「秒」。
                        如果換成 Whisper 或其他語音模型，因為大腦的壓縮比例不同，這個換算公式的數字就會完全不一樣了！
                        """
                        ts_left = max((_start*60-30)/1000, 0) 
                        ts_right = min((_end*60-30)/1000, (ts_max*60-30)/1000)
                        timestamp.append([tokens[token_id], ts_left, ts_right]) # 把剛剛對齊好的純文字字元 (tokens[token_id])，跟算出來的開始與結束秒數綁定在一起，存進最後的輸出列表裡，然後繼續往下一個字前進。
                        token_id += 1
                    _start = _end

                result_i = {"key": key[i], "text": text, "timestamp": timestamp} # 把音檔名稱 (key)、翻譯好的文字 (text)，以及剛剛精算出的時間戳記 (timestamp) 全部包在一起。
                results.append(result_i)
            else:
                result_i = {"key": key[i], "text": text} # 打包音檔名稱 (keSenseVoiceEncoderSmally) 和文字 (text)。
                results.append(result_i)
        return results, meta_data

    # 匯出成 ONNX 或 Libtorch 格式
    def export(self, **kwargs):
        from export_meta import export_rebuild_model

        if "max_seq_len" not in kwargs:
            kwargs["max_seq_len"] = 512 # 匯出靜態圖（Static Graph）時常見的設定。如果沒有指定，它會預設輸入的最大序列長度為 512，確保底層運算時的記憶體分配是固定的。
        models = export_rebuild_model(model=self, **kwargs) # 把當前複雜的 PyTorch 模型打包並稍微改寫結構，轉換成 ONNX 或 TorchScript 這種容易被其他程式語言讀取執行的格式。
        return models
