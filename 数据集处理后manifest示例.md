明确提供：
#### 1. 有序通道名。
#### 2. channel_map_id：“EEG channel index → 通道名称”的映射
```
eeg 第 j 个通道
→ eeg_channel_indices[j]
→ channel catalog 中对应的名称
→ 再由模型侧映射到标准 64 导
```
例如：
```
channel_catalog = ["Cz", "Fz", "Fp1", "F7"]
eeg_channel_indices = [2, 0, 3]
```
那么：
```
eeg[0] → catalog[2] → Fp1
eeg[1] → catalog[0] → Cz
eeg[2] → catalog[3] → F7
```

#### 3. 原始以及所有处理做完后的采样率和窗口长度。
#### 4. 是否已经做 waveform z-score。
#### 5. padding 和 valid_samples 的精确定义。

`padding` 的精确定义：

当前数据 metadata 声明的是：`normalize-real-samples-then-right-zero-pad`

其预期含义是：
假设 canonical 输出固定为：100 Hz × 10 秒 = 1000 点，某条数据只有 N 个真实有效点，则：

```
原始有效数据
[C,N]

→ 只使用这 N 个真实点计算每通道 mean/std
→ 对 [C,N] 做 z-score
→ 在右侧补 1000-N 个 0
→ [C,1000]
```

例如 valid_samples=700：
```
waveform[:, 0:700]      = 标准化后的真实 EEG
waveform[:, 700:1000]   = 0
```
这和“先补零再做标准化”不同：
```
正确：
真实数据标准化 → 补零
补零部分最终仍为 0

错误：
真实数据补零 → 对整个 1000 点标准化
补零部分会变成非零常数
```

`valid_samples` 的精确定义
理想定义应是：在 resample 到 canonical 100 Hz 后，当前固定窗口中从时间轴起点开始连续有效的真实 EEG 点数，不包含右侧 padding。

因此：
```
valid_samples = 1000
→ 完整 10 秒，无时间 padding

valid_samples = 700
→ 前 7 秒是真实数据，后 3 秒是 padding

valid_samples = 400
→ 前 4 秒是真实数据，后 6 秒是 padding
```

它是 row-level 字段，因此通常表示所有现存通道共享同一个时间有效长度。
通道是否存在则由另一个字段表达：
`valid_channel_mask_json`

也就是两个 mask 管不同维度：
```
valid_samples
→ 时间轴有多少真实点

valid_channel_mask_json
→ 哪些通道真实存在/有效
```


#### 6. full/partial window 数量和 NaN/Inf QC。
推荐的轻量NaN/Inf QC 格式：
```
{
  "finite_qc": {
    "passed": true,

    "total_value_count": 320000000,
    "nan_value_count": 0,
    "posinf_value_count": 0,
    "neginf_value_count": 0,
    "nonfinite_value_count": 0,

    "rows_with_nan": 0,
    "rows_with_inf": 0,
    "rows_with_nonfinite": 0,

    "channels_with_nonfinite": 0
  }
}
```

#### 7. schema、recipe、channel catalog 的稳定 hash。