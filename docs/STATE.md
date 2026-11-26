# pretrain V2 状态

这是给人看的极简状态表，只回答：哪些数据集已处理、最终 Lance 在哪里、形状是什么、还有哪些未完成。详细 QC 和错误证据不堆在这里，沿 `run` 路径查看。

范围：迁移后的 205 个数据集。截至 2026-11-25，只读 COS 审计确认 **96/205 已发布，109/205 尚无满足本表门禁的权威 V2 release**。完整 205 行状态、阻断原因和路径见 [205数据集当前状态.csv](../manifests/205数据集当前状态.csv)。

状态含义：`PUBLISHED` 使用当前合同并通过 acceptance、正式训练读回和 `_SUCCESS`；`PUBLISHED_LEGACY` 通过 acceptance/publication/正式训练读回，但旧合同没有 `_SUCCESS`；`PUBLISHED_PARTIAL` 明确保留了错误 recording 证据，只发布 accepted 范围；其他状态均不算处理完成。

| dataset | 状态 | 范围 | Lance 形状 | 窗口数 | 当前权威 run | 最终 Lance / 下一步 |
|---|---|---|---|---:|---|---|
| bbbd | FAILED | 0/1158 under fixed V2 recipe | `[channels,2000]; tail [channels,<2000]` | 0 | `run-20261125T074800Z-bbbd` | TIONE `train-1725443479054966656`；1158/1158 均为 128 Hz，Nyquist 64 Hz 无法满足固定 1–75 Hz；无 release，DEATH_SENTENCE |
| ds002718 | SOURCE_GATE | legacy complete preserved | `[channels,2000]; tail [channels,<2000]` | 5351 | `—` | TIONE `train-1724201821829990144`；SOURCE_GATE |
| ds002722 | PUBLISHED | complete | `[channels,2000]; tail [channels,<2000]` | 8281 | `qc-v022-20261124T075946Z` | cos://omni-eeg-processed-1442740494/datasets/ds002722/releases/pretrain_V2/v022/data.lance/ |
| ds002723 | PUBLISHED | complete | `[channels,2000]; tail [channels,<2000]` | 3789 | `run-20261125T075548Z-ds002723` | cos://omni-eeg-processed-1442740494/datasets/ds002723/releases/pretrain_V2/r001/data.lance/ |
| ds002724 | PUBLISHED | complete | `[channels,2000]; tail [channels,<2000]` | 10345 | `run-20261125T075730Z-ds002724` | cos://omni-eeg-processed-1442740494/datasets/ds002724/releases/pretrain_V2/r001/data.lance/ |
| ds002778 | SOURCE_GATE | legacy complete preserved | `[channels,2000]; tail [channels,<2000]` | 928 | `—` | TIONE `train-1724195036788960000`；SOURCE_GATE |
| ds002893 | PUBLISHED | complete | `[channels,2000]; tail [channels,<2000]` | 13628 | `pretrain-v2-direct-v016` | cos://omni-eeg-processed-1442740494/datasets/ds002893/releases/pretrain_V2/v016/data.lance/ |
| ds003004 | PUBLISHED_PARTIAL | partial 29/34 | `[channels,2000]; tail [channels,<2000]` | 15250 | `run-20261125T083032Z-ds003004` | cos://omni-eeg-processed-1442740494/datasets/ds003004/releases/pretrain_V2_partial/r001/data.lance/ |
| ds003638 | PUBLISHED | complete 57/57 | `[channels,2000]; tail [channels,<2000]` | 14641 | `run-20261125T084219Z-ds003638` | cos://omni-eeg-processed-1442740494/datasets/ds003638/releases/pretrain_V2/r001/data.lance/ |
| ds003655 | PUBLISHED | complete 156/156 | `[channels,2000]; tail [channels,<2000]` | 47200 | `run-20261125T082207Z-ds003655` | cos://omni-eeg-processed-1442740494/datasets/ds003655/releases/pretrain_V2/r001/data.lance/ |
| ds003801 | PUBLISHED | complete | `[channels,2000]; tail [channels,<2000]` | 4933 | `run-20261125T074200Z-ds003801` | cos://omni-eeg-processed-1442740494/datasets/ds003801/releases/pretrain_V2/r001/data.lance/ |
| ds004147 | PUBLISHED | complete 12/12 | `[channels,2000]; tail [channels,<2000]` | 3466 | `run-20261125T073817Z-ds004147` | cos://omni-eeg-processed-1442740494/datasets/ds004147/releases/pretrain_V2/r001/data.lance/ |
| ds004262 | PUBLISHED | complete | `[channels,2000]; tail [channels,<2000]` | 3016 | `—` | cos://omni-eeg-processed-1442740494/datasets/ds004262/releases/pretrain_V2/v016/data.lance/ |
| ds004357 | PUBLISHED_LEGACY | complete | `[channels,2000]; tail [channels,<2000]` | 4077 | `—` | cos://omni-eeg-processed-1442740494/datasets/ds004357/releases/pretrain_V2/v007/data.lance/ |
| ds004660 | PUBLISHED | complete | `[channels,2000]; tail [channels,<2000]` | 8585 | `run-20261125T074342Z-ds004660` | cos://omni-eeg-processed-1442740494/datasets/ds004660/releases/pretrain_V2/r001/data.lance/ |
| ds005397 | FAILED | complete candidate 26/26 | `[channels,2000]; tail [channels,<2000]` | 待验收 | `run-20261125T083410Z-ds005397` | TIONE `train-1725466725660624640`；FAILED：Current rounded-clock run failed: event sample/onset clock differs despite one-sample contract; no r001 release; legacy v001 preserved |
| ds005505 | PUBLISHED | complete 1342/1342 | `[channels,2000]; tail [channels,<2000]` | 42959 | `run-20261125T072954Z-ds005505` | cos://omni-eeg-processed-1442740494/datasets/ds005505/releases/pretrain_V2/r001/data.lance/ |
| ds006126 | PUBLISHED | complete | `[channels,2000]; tail [channels,<2000]` | 3914 | `run-20261125T074828Z-ds006126` | cos://omni-eeg-processed-1442740494/datasets/ds006126/releases/pretrain_V2/r001/data.lance/ |
| ds006222 | PUBLISHED | complete | `[channels,2000]; tail [channels,<2000]` | 19036 | `run-20261125T082013Z-ds006222` | cos://omni-eeg-processed-1442740494/datasets/ds006222/releases/pretrain_V2/r001/data.lance/ |
| ds006839 | PUBLISHED | complete | `[channels,2000]; tail [channels,<2000]` | 9655 | `run-20261125T080720Z-ds006839` | cos://omni-eeg-processed-1442740494/datasets/ds006839/releases/pretrain_V2/r001/data.lance/ |
| ds007180 | PUBLISHED | complete | `[channels,2000]; tail [channels,<2000]` | 12513 | `run-20261125T075004Z-ds007180` | cos://omni-eeg-processed-1442740494/datasets/ds007180/releases/pretrain_V2/r001/data.lance/ |
| ds007526 | PUBLISHED | complete | `[channels,2000]; tail [channels,<2000]` | 7228 | `run-20261125T081824Z-ds007526` | cos://omni-eeg-processed-1442740494/datasets/ds007526/releases/pretrain_V2/r001/data.lance/ |
| ds007648 | PUBLISHED_PARTIAL | partial | `[channels,2000]; tail [channels,<2000]` | 5598 | `qc-v023-20261124T075949Z` | cos://omni-eeg-processed-1442740494/datasets/ds007648/releases/pretrain_V2_partial/v023/data.lance/ |
| ds007720 | PUBLISHED | complete | `[channels,2000]; tail [channels,<2000]` | 11604 | `run-20261125T081621Z-ds007720` | cos://omni-eeg-processed-1442740494/datasets/ds007720/releases/pretrain_V2/r001/data.lance/ |
| ds007865 | PUBLISHED | complete | `[channels,2000]; tail [channels,<2000]` | 3691 | `run-20261125T075235Z-ds007865` | cos://omni-eeg-processed-1442740494/datasets/ds007865/releases/pretrain_V2/r001/data.lance/ |
| ds007895 | PUBLISHED | complete | `[channels,2000]; tail [channels,<2000]` | 6497 | `run-20261125T080512Z-ds007895` | cos://omni-eeg-processed-1442740494/datasets/ds007895/releases/pretrain_V2/r001/data.lance/ |
| ds008083 | SOURCE_GATE | legacy complete preserved | `[channels,2000]; tail [channels,<2000]` | 15173 | `—` | TIONE `train-1724204485875730176`；SOURCE_GATE |
| ds008115 | PUBLISHED | complete | `[channels,2000]; tail [channels,<2000]` | 9720 | `run-20261125T082636Z-ds008115` | cos://omni-eeg-processed-1442740494/datasets/ds008115/releases/pretrain_V2/r001/data.lance/ |
| ds004015 | PUBLISHED | complete | `[channels,2000]; tail [channels,<2000]` | 17040 | `run-20261124T121214Z-ds004015` | cos://omni-eeg-processed-1442740494/datasets/ds004015/releases/pretrain_V2/r002/data.lance/ |
| ds004121 | PUBLISHED | complete 60/60 | `[channels,2000]; tail [channels,<2000]` | 11225 | `run-20261124T123122Z-ds004121` | cos://omni-eeg-processed-1442740494/datasets/ds004121/releases/pretrain_V2/r001/data.lance/ |
| ds004122 | PUBLISHED | complete 63/63 | `[channels,2000]; tail [channels,<2000]` | 16569 | `run-20261124T123137Z-ds004122` | cos://omni-eeg-processed-1442740494/datasets/ds004122/releases/pretrain_V2/r001/data.lance/ |
| ds004105 | PUBLISHED | complete 34/34 | `[channels,2000]; tail [channels,<2000]` | 9441 | `run-20261124T122726Z-ds004105` | cos://omni-eeg-processed-1442740494/datasets/ds004105/releases/pretrain_V2/r001/data.lance/ |
| ds004117 | PUBLISHED | complete 85/85 | `[channels,2000]; tail [channels,<2000]` | 5606 | `run-20261124T123644Z-ds004117` | cos://omni-eeg-processed-1442740494/datasets/ds004117/releases/pretrain_V2/r001/data.lance/ |
| ds004603 | PUBLISHED | complete 37/37 | `[channels,2000]; tail [channels,<2000]` | 11050 | `run-20261124T141716Z-ds004603` | cos://omni-eeg-processed-1442740494/datasets/ds004603/releases/pretrain_V2/r001/data.lance/ |
| ds004152 | PUBLISHED | complete 21/21 | `[channels,2000]; tail [channels,<2000]` | 4134 | `run-20261124T142621Z-ds004152` | cos://omni-eeg-processed-1442740494/datasets/ds004152/releases/pretrain_V2/r001/data.lance/ |
| ds004264 | PUBLISHED | complete 21/21 | `[channels,2000]; tail [channels,<2000]` | 2695 | `run-20261124T142631Z-ds004264` | cos://omni-eeg-processed-1442740494/datasets/ds004264/releases/pretrain_V2/r001/data.lance/ |
| ds005594 | PUBLISHED | complete 16/16 | `[channels,2000]; tail [channels,<2000]` | 4438 | `run-20261124T143057Z-ds005594` | cos://omni-eeg-processed-1442740494/datasets/ds005594/releases/pretrain_V2/r001/data.lance/ |
| ds005121 | PUBLISHED | complete 39/39 | `[channels,2000]; tail [channels,<2000]` | 14605 | `run-20261124T143748Z-ds005121` | cos://omni-eeg-processed-1442740494/datasets/ds005121/releases/pretrain_V2/r001/data.lance/ |
| ds005095 | PUBLISHED | complete 48/48 | `[channels,2000]; tail [channels,<2000]` | 6110 | `run-20261124T145005Z-ds005095` | cos://omni-eeg-processed-1442740494/datasets/ds005095/releases/pretrain_V2/r001/data.lance/ |
| ds004284 | PUBLISHED | complete 18/18 | `[channels,2000]; tail [channels,<2000]` | 3413 | `run-20261124T152725Z-ds004284` | cos://omni-eeg-processed-1442740494/datasets/ds004284/releases/pretrain_V2/r001/data.lance/ |
| ds004561 | PUBLISHED | complete 23/23 | `[channels,2000]; tail [channels,<2000]` | 4108 | `run-20261124T152735Z-ds004561` | cos://omni-eeg-processed-1442740494/datasets/ds004561/releases/pretrain_V2/r001/data.lance/ |
| ds004572 | PUBLISHED | complete 516/516 | `[channels,2000]; tail [channels,<2000]` | 19428 | `run-20261124T152853Z-ds004572` | cos://omni-eeg-processed-1442740494/datasets/ds004572/releases/pretrain_V2/r001/data.lance/ |
| ds007647 | PUBLISHED | complete 40/40 | `[channels,2000]; tail [channels,<2000]` | 5619 | `run-20261124T172118Z-ds007647` | cos://omni-eeg-processed-1442740494/datasets/ds007647/releases/pretrain_V2/r001/data.lance/ |
| ds007968 | PUBLISHED | complete 40/40 | `[channels,2000]; tail [channels,<2000]` | 4744 | `run-20261124T172422Z-ds007968` | cos://omni-eeg-processed-1442740494/datasets/ds007968/releases/pretrain_V2/r001/data.lance/ |
| ds005841 | PUBLISHED_PARTIAL | partial 257/288 | `[channels,2000]; tail [channels,<2000]` | 7010 | `run-20261124T173517Z-ds005841` | cos://omni-eeg-processed-1442740494/datasets/ds005841/releases/pretrain_V2_partial/r001/data.lance/ |
| ds004816 | PUBLISHED | complete 20/20 | `[channels,2000]; tail [channels,<2000]` | 3867 | `run-20261124T174515Z-ds004816` | cos://omni-eeg-processed-1442740494/datasets/ds004816/releases/pretrain_V2/r001/data.lance/ |
| ds004817 | PUBLISHED_PARTIAL | partial 19/20 | `[channels,2000]; tail [channels,<2000]` | 4092 | `run-20261124T174747Z-ds004817` | cos://omni-eeg-processed-1442740494/datasets/ds004817/releases/pretrain_V2_partial/r001/data.lance/ |
| ds004582 | PUBLISHED | complete 73/73 | `[channels,2000]; tail [channels,<2000]` | 12367 | `run-20261124T175356Z-ds004582` | cos://omni-eeg-processed-1442740494/datasets/ds004582/releases/pretrain_V2/r001/data.lance/ |
| ds004706 | RUNNING | partial 283/298 | `[channels,2000]; tail [channels,<2000]` | 待验收 | `run-20261125T021920Z-ds004706` | TIONE `train-1725278025886598912`；RUNNING |
| ds003846 | PUBLISHED | complete 50/50 | `[channels,2000]; tail [channels,<2000]` | 8203 | `run-20261125T022825Z-ds003846` | cos://omni-eeg-processed-1442740494/datasets/ds003846/releases/pretrain_V2/r001/data.lance/ |
| ds004369 | PUBLISHED | complete 41/41 | `[4,2000]` | 13440 | `run-20261125T055342Z-ds004369` | cos://omni-eeg-processed-1442740494/datasets/ds004369/releases/pretrain_V2/r001/data.lance/ |
| ds004951 | PUBLISHED_PARTIAL | partial 22/23 | `[channels,2000]; tail [channels,<2000]` | 8896 | `run-20261125T091626Z-ds004951` | cos://omni-eeg-processed-1442740494/datasets/ds004951/releases/pretrain_V2_partial/r001/data.lance/ |
| ds004980 | PUBLISHED | complete 17/17 | `[channels,2000]; tail [channels,<2000]` | 13276 | `run-20261125T032134Z-ds004980` | cos://omni-eeg-processed-1442740494/datasets/ds004980/releases/pretrain_V2/r002/data.lance/ |
| ds005131 | PUBLISHED | complete 63/63 | `[channels,2000]; tail [channels,<2000]` | 18769 | `run-20261125T033317Z-ds005131` | cos://omni-eeg-processed-1442740494/datasets/ds005131/releases/pretrain_V2/r002/data.lance/ |
| ds003039 | PUBLISHED_PARTIAL | partial 16/19 | `[channels,2000]; tail [channels,<2000]` | 5291 | `run-20261125T034256Z-ds003039` | cos://omni-eeg-processed-1442740494/datasets/ds003039/releases/pretrain_V2_partial/r001/data.lance/ |
| ds005363 | PUBLISHED_PARTIAL | partial 41/43 | `[channels,2000]; tail [channels,<2000]` | 14846 | `run-20261125T032147Z-ds005363` | cos://omni-eeg-processed-1442740494/datasets/ds005363/releases/pretrain_V2_partial/r002/data.lance/ |
| ds004256 | PUBLISHED | complete 53/53 | `[channels,2000]; tail [channels,<2000]` | 15273 | `run-20261125T033330Z-ds004256` | cos://omni-eeg-processed-1442740494/datasets/ds004256/releases/pretrain_V2/r002/data.lance/ |
| ds005811 | PUBLISHED | complete 448/448 | `[channels,2000]; tail [channels,<2000]` | 8821 | `run-20261125T034849Z-ds005811` | cos://omni-eeg-processed-1442740494/datasets/ds005811/releases/pretrain_V2/r001/data.lance/ |
| ds006434 | PUBLISHED_PARTIAL | partial 50/118 | `[channels,2000]; tail [channels,<2000]` | 30670 | `run-20261125T033924Z-ds006434` | cos://omni-eeg-processed-1442740494/datasets/ds006434/releases/pretrain_V2_partial/r001/data.lance/ |
| ds004579 | PUBLISHED_PARTIAL | partial 134/139 | `[channels,2000]; tail [channels,<2000]` | 19257 | `run-20261125T041928Z-ds004579` | cos://omni-eeg-processed-1442740494/datasets/ds004579/releases/pretrain_V2_partial/r001/data.lance/ |
| first-episode-psychosis-rest-1 | FAILED | complete candidate 82/82 | `[channels,2000]; tail [channels,<2000]` | 未发布 | `run-20261125T035543Z-first-episode-psychosis-rest-1` | TIONE `train-1725326572251373440`；仍在 `sub-1448` 报 invalid BrainVision MarkerFile；`code-20261125-48` 未贯通有效 reader 路径，不自动重试 |
| m3cv | PUBLISHED | complete 2469/2469 | `[channels,2000]; tail [channels,<2000]` | 36,746；1,147 尾窗 | `run-20261125T033256Z-m3cv` | `r001`；acceptance、publication、训练读回、Lance 与 `_SUCCESS` 全通过；cos://omni-eeg-processed-1442740494/datasets/m3cv/releases/pretrain_V2/r001/data.lance/ |
| le-petit-prince-multi-talker | PUBLISHED_PARTIAL | partial 25/26 | `[channels,2000]; tail [channels,<2000]` | 7,157；24 尾窗 | `run-20261125T070320Z-le-petit-prince-multi-talker` | `r001`；`sub-15` 因 `SOURCE_BINARY_GRID_MISALIGNED` 隔离；acceptance、publication、首/中/尾训练读回、Lance 与 `_SUCCESS` 全部一致；cos://omni-eeg-processed-1442740494/datasets/le-petit-prince-multi-talker/releases/pretrain_V2_partial/r001/data.lance/ |
| emd-eeg-moments | PUBLISHED_PARTIAL | partial 767/768 | `[channels,2000]; tail [channels,<2000]` | 35,366；766 尾窗 | `run-20261125T033729Z-emd-eeg-moments` | `r001`；1 条 grid quarantine；全部发布门禁通过；cos://omni-eeg-processed-1442740494/datasets/emd-eeg-moments/releases/pretrain_V2_partial/r001/data.lance/ |

`ds004015` 的历史 TIONE `train-1724740147360836352` 在写 acceptance 时因 NumPy `int64` JSON 序列化失败；旧 `r001` 下约 2.63 GB 对象仍是不完整候选，保留且不覆盖。修复后的 `run-20261124T121214Z-ds004015`（TIONE `train-1724851632024706944`，`code-20261124-03`）在允许的 `tione_group` 完成 36/36 全波形 QC、17,040 窗构建、本地验收、COS 复制校验和 17,040/17,040 正式训练读回。`r002` acceptance、publication 与 `_SUCCESS` 一致，Lance version 267，现为 `PUBLISHED`。训练读回证据嵌在 publication，尚无独立 `training_readback.json`。

`ds002893` 对应 TIONE `train-1724679255592656768` 已完成 52/52 recording、13,628 窗的完整范围 V2 发布；release `v016` 的 acceptance、publication 与 `_SUCCESS` 均存在，publication 记录 Lance version 213 和正式训练读回 13,628/13,628。该历史合同把训练读回证据嵌在 `publication.json`，没有独立 `training_readback.json`；这是 artifact 合同缺口，不否定已实际通过的读回。

## 本轮 agent 门禁结果

- `ds004574`：146/146 的 500 Hz、逐 recording 60 Hz、63 路 EEG 冻结子集、MNE EEGLAB 单位合同、SET/FDT grid、finite/nonzero 和 103,576 个事件 origin=0/bounds 均可闭合；但 `sub-003` 与 `sub-025` 各有一个 duration=`NaN` 的内部 EEGLAB boundary。当前 transform/validator 会跨 discontinuity 滤波和切窗，该原因又不在批准的 partial 白名单，保持 `BLOCKED/EEGLAB_INTERNAL_DISCONTINUITY_ADAPTER_MISSING`，未提交。
- `ds008711`：64/64 continuous SET/FDT sidecar 可读且均为 500 Hz，但 64/64 的 32 个通道 type 均为 `N/A`，EEG/辅助通道身份、物理单位、逐 recording 工频、事件时钟和完整波形 QC 均未闭合；保持 `SOURCE_GATE/EVIDENCE_MISSING`，未提交。
- `ds004541`：9/9 native metadata 可读，但 9/9 工频未解决；保持 `SOURCE_GATE`，未提交。
- `ds004661`：17 条均为 128 Hz，Nyquist 64 Hz，不能满足固定 75 Hz 上限；不提交。
- `ds004105`：34/34 唯一 recording 的完整波形 finite，metadata/event、1024 Hz 与逐 recording 60 Hz 证据齐全；旧 waveform JSONL 的第 35 行是同一 recording 网络失败后的成功重试。根级 BIDS EEG JSON 已作为显式、经过审核的 `eeg_json` 路径冻结进 manifest，不做模糊继承搜索；TIONE `train-1724859305076055936` 已提交，当前保持 `RUNNING`，等待完整 V2 发布门禁。
- `ds004121`：60/60 完整波形 finite，metadata、1024 Hz、64 EEG、`uV`、60 Hz 和事件边界均通过；根级 BIDS EEG JSON 已显式冻结进 manifest，原 `ADAPTER_ERROR` 提交阻塞关闭。TIONE `train-1724861288666639104` 已提交，当前保持 `RUNNING`，等待完整 V2 发布门禁。
- `ds004122`：63/63 完整波形 finite，metadata、1024 Hz、64 EEG、`uV`、60 Hz 和事件边界均通过；根级 BIDS EEG JSON 已显式冻结进 manifest，原 `ADAPTER_ERROR` 提交阻塞关闭。TIONE `train-1724861404727861120` 已提交，当前保持 `RUNNING`，等待完整 V2 发布门禁。
- `ds004603`：raw BrainVision 权威范围为 37/37；1024 Hz、逐 recording 50 Hz、64 EEG + 1 TRIG、EEG 单位 `µV`、156,876 个事件及 sample origin 0 均闭合，37/37 完整波形 finite/nonzero。已冻结并远端逐字节验证 `code-20261124-09`，TIONE `train-1724914585088417664` 已提交到 `tione_group`，候选 `r001`，等待后续发布验收。
- `ds007647`：40/40 BrainVision 冻结 VHDR 合同已闭合：32 个 `µV` EEG、1000 Hz、60 Hz、INT16 grid、63,777 个 origin=0 事件均通过；原 `channels.tsv` type/units=`n/a` 保留不改。80/80 tests，`code-20261125-10`；已提交 `train-1725007221152155392` / `run-20261124T172118Z-ds007647` 到 `tione_group`，当前 `PENDING/ResourceInsufficient`。
- `ds004579`：`CODE_FIXED(code-20261125-53) → SUBMITTED → PUBLISHED_PARTIAL`。冻结 MNE EEGLAB `CAL=1e-6` 读取合同；精确隔离 4 条 event-duration 越界与 `sub-062` X/Y/Z 通道身份冲突。TIONE `train-1725338511307713280` / `run-20261125T041928Z-ds004579` 正式发布 partial `r001`：134/139、19,257 窗、134 个尾窗；acceptance、publication、19,257/19,257 训练读回、Lance 和 `_SUCCESS` 一致。最终 Lance：`cos://omni-eeg-processed-1442740494/datasets/ds004579/releases/pretrain_V2_partial/r001/data.lance/`。
- `ds004580`：`PUBLISHED_PARTIAL`。`train-1725367116502936320` / `run-20261125T051619Z-ds004580` 正式发布 `pretrain_V2_partial/r001`：146/147 recordings、13,096 windows、146 个变长尾窗；`sub-033` 因 2 条 duration 越界事件精确隔离。acceptance、publication、13,096/13,096 训练读回、Lance 和 `_SUCCESS` 一致。最终 Lance：`cos://omni-eeg-processed-1442740494/datasets/ds004580/releases/pretrain_V2_partial/r001/data.lance/`。
- `ds005089`：36/36 完整波形 finite（33 条普通 SET/FDT + 3 条 MATLAB v5 ALLEEG）；正式流程缺 ALLEEG structured reader，且逐 recording 工频、物理单位和事件 origin 尚未冻结，保持 `SOURCE_GATE/ADAPTER_ERROR+EVIDENCE_MISSING`。
- `ds004515`：54/54 SET/FDT 对象网格、500/60 Hz、39,621 个事件边界和 0 内部 discontinuity 均闭合；但 48/54 的 `Below Eye`、`Above Eye`、双 mastoid 与末三辅助行均无坐标，JSON 同时声明 64 EEG、EOG=0、Misc=66。无法不猜测地冻结 EEG-only 身份/单位，且证据缺口不能 partial，保持 `BLOCKED/CHANNEL_IDENTITY_AND_UNIT_EVIDENCE_MISSING`，未提交。
- `ds003519`：`PUBLISHED r001`。54/54 recordings、7,408 windows、54 个变长尾窗、0 quarantine；acceptance、publication、7,408/7,408 远程读回、Lance 和 `_SUCCESS` 一致。正式路径：`cos://omni-eeg-processed-1442740494/datasets/ds003519/releases/pretrain_V2/r001/data.lance/`。
- `ds006367`：52/52 为 1000 Hz epoched SET/FDT，事件 origin candidates 为 0；工频、逐行通道类型/单位、完整波形及 epoched adapter 合同未闭合，保持 `SOURCE_GATE/EVIDENCE_MISSING+ADAPTER_ERROR`。
- `ds006370`：56/56 为 1000 Hz epoched SET/FDT，事件 origin candidates 为 0；工频、逐行通道类型/单位、完整波形及 epoched adapter 合同未闭合，保持 `SOURCE_GATE/EVIDENCE_MISSING+ADAPTER_ERROR`。
- `ds006554`：`PUBLISHED_PARTIAL`。`train-1725370834166229760` / `run-20261125T052342Z-ds006554` 正式发布 `pretrain_V2_partial/r001`：43/47 recordings、9,357 windows、43 个变长尾窗；4 条确定性 SET/FDT grid 错误精确隔离。acceptance、publication、9,357/9,357 训练读回、Lance 和 `_SUCCESS` 一致。最终 Lance：`cos://omni-eeg-processed-1442740494/datasets/ds006554/releases/pretrain_V2_partial/r001/data.lance/`。
- `ds006923`：280/280 exact EEG JSON 均为 128 Hz、60 Hz、128 EEG；源范围为 140 条 epoched SET + 140 条 continuous SET。128 Hz 的 Nyquist 仅 64 Hz，无法执行不可变的 1–75 Hz V2 recipe；accepted scope 0/280，无合法 partial，保持 `SOURCE_GATE/BLOCKED`，未提交。
- `ds007069`：281/281 continuous SET 均为 500 Hz/60 Hz，JSON 声明 29 EEG + 3 EOG；但 281 份 channels.tsv 的 8,992 行 type/units 全为 `n/a`，无法权威冻结逐行 EEG/EOG 身份和物理单位。`sub-1081` 唯一缺 events.tsv，证据缺失不属于 partial 白名单；accepted scope 0/281，保持 `SOURCE_GATE/BLOCKED`，未提交。
- `ds007052`：288 条 embedded SET 的 JSON 均声明 29 EEG + 3 EOG、500 Hz/60 Hz，但 9,216 个 TSV type/unit 以及 SET `chanlocs.type` 全为空/`n/a`，不能按名称映射 EEG/EOG 或猜单位。`sub-2002/sub-2025/sub-3042` 又各只有 32×25 samples（0.05 秒）且无事件；全局 evidence gap 不能用 partial 绕过。保持 `SOURCE_GATE/BLOCKED`，未提交。
- `ds004369`：`PUBLISHED`。`train-1725385939373271808` / `run-20261125T055342Z-ds004369` 正式发布 `pretrain_V2/r001`：complete 41/41 recordings、13,440 windows、0 个尾窗，shape `[4,2000]`。acceptance、publication、13,440/13,440 正式训练读回、Lance 和 `_SUCCESS` 一致。最终 Lance：`cos://omni-eeg-processed-1442740494/datasets/ds004369/releases/pretrain_V2/r001/data.lance/`。
- `ds005841`：新 BDF header/object-grid 门禁纠正旧摘要，31/288 条 COS 对象小于 header 声明长度，按 `SOURCE_BINARY_TRUNCATED` 精确隔离；257/288 使用冻结的 64 头皮 EEG（排除 EXG1–8/Status）合同。86/86 tests，`code-20261125-12`；已提交 `train-1725014265125013248` / `run-20261124T173517Z-ds005841`，当前 `PENDING/ResourceInsufficient`，不是失败，不停止或重提。

- 2026-11-25 实时调度快照：状态表内 28 个明确任务逐项查询后，`ds004105`、`ds004121`、`ds004122` 为 `RUNNING`，其余 25 个为 `PENDING/ResourceInsufficient`。该 snapshot 取代更早的 `FIFOBlocked`、`SUBMITTING` 或“reason 未返回”实时描述；历史记录仍保留。调度状态不等于发布状态。
- `ds006547`：31/31 BrainVision 严格冻结 VHDR 通道合同闭合：63 个 `µV` EEG + 4 个 `C` 辅助，500/60 Hz，21,615 events origin=0/bounds 与 grid 全通过；JSON 64 EEG/FCz 冲突原样保留。75/75 tests，`code-20261125-05`。已提交 `tione_group`：task `train-1724994335235245824`，run `run-20261124T165542Z-ds006547`，候选 `r001`；当前 `PENDING/FIFOBlocked`。run：`cos://omni-eeg-processed-1442740494/datasets/ds006547/runs/run-20261124T165542Z-ds006547/`。
- `ds007471`：32/32 hyperscanning BrainVision 冻结合同闭合：54 EEG + 8 EOG + 2 reference，INT16、1000/60 Hz、70,348 无-sample事件 bounds、partner alias/grid 全通过；L/R 冲突原样保留。76/76 tests，`code-20261125-08`。已提交 `tione_group`：task `train-1725002947115100928`，run `run-20261124T171248Z-ds007471`，候选 `r001`；当前 `PENDING`，reason 尚未返回。run：`cos://omni-eeg-processed-1442740494/datasets/ds007471/runs/run-20261124T171248Z-ds007471/`。
- `ds004166`：213 组 SET/FDT、约 83.07 GB，但 EEG JSON/channels/events sidecar 全部为零，采样率、工频、通道/单位、事件时钟及完整波形均未冻结，保持 `SOURCE_GATE/EVIDENCE_MISSING+ADAPTER_ERROR`。
- `ds006260`：366/366 embedded SET 为 256 Hz、60 Hz，但通道 type/units 全为 `n/a`，145 条缺 events TSV，且无完整波形证据，保持 `SOURCE_GATE/EVIDENCE_MISSING+ADAPTER_ERROR`。
- `ds004860`：31/31 BDF 明确 50 Hz、header 网格对齐；无 `channels.tsv`，事件 onset 实为毫秒且两条存在明确 sample/clock 差异，无完整波形 QC，保持 `SOURCE_GATE/EVIDENCE_MISSING+METADATA_ERROR+ADAPTER_ERROR`。
- `ds005296`：62/62 共享一个 32 路 SET/TSV 名称签名，但 channels 的 type/unit 全为 `n/a`，JSON 不声明 EEG/EOG/Misc 数量，SET `chanlocs.type` 也为空；`LE/HE/A2` 身份和物理单位无法不猜测地冻结，数据集级 evidence gap 不能 partial，保持 `BLOCKED/CHANNEL_IDENTITY_AND_UNIT_EVIDENCE_MISSING`，未提交。
- `ds005688`：89/89 BrainVision 配对、超高采样率和逐 recording 工频通过，但 header 仅 1 通道而 86 条 JSON 声明 4 EEG + 1 EOG；无 channels/events sidecar 和全波形 QC，保持 `SOURCE_GATE/METADATA_ERROR+EVIDENCE_MISSING+ADAPTER_ERROR`。
- `ds005863`：357/357 BrainVision 严格 VHDR/partner-alias 合同闭合：26 stored EEG + 3 EOG + 2 MISC，JSON 27 EEG 含未存储 Cz reference；500/60 Hz、191,735 events origin=0/bounds、float32 grid 全通过。76/76 tests，`code-20261125-06`。已提交 `tione_group`：task `train-1724997688891672320`，run `run-20261124T170221Z-ds005863`，候选 `r001`；当前 `PENDING/FIFOBlocked`。run：`cos://omni-eeg-processed-1442740494/datasets/ds005863/runs/run-20261124T170221Z-ds005863/`。
- `ds006018`：357/357 BrainVision 严格冻结 VHDR 合同闭合：26 stored EEG + 3 EOG + 2 MISC，JSON 27 EEG 含未存储 Cz reference；partners、float32 grid、500/60 Hz、191,735 events origin=0/bounds 与既有 full-wave evidence 通过。76/76 tests，`code-20261125-07`。已提交 `tione_group`：task `train-1724999918882774912`，run `run-20261124T170647Z-ds006018`，候选 `r001`；当前 `PENDING/FIFOBlocked`。run：`cos://omni-eeg-processed-1442740494/datasets/ds006018/runs/run-20261124T170647Z-ds006018/`。
- `ds005565`：24/24 continuous SET/FDT 为 500 Hz、60 Hz；唯一 32 路签名的 type/unit 全为 `n/a`，JSON 不声明 EEG/EOG/Misc 数量，SET `chanlocs.type` 为空，`LE/HE/A2` 身份和单位无法不猜测地冻结。数据集级 evidence gap 不能 partial，保持 `BLOCKED/CHANNEL_IDENTITY_AND_UNIT_EVIDENCE_MISSING`，未提交。
- `ds005866`：60/60 continuous SET/FDT 为 500 Hz、60 Hz；唯一 32 路签名的 type/unit 全为 `n/a`，JSON 不声明 EEG/EOG/Misc 数量，SET `chanlocs.type` 为空，`LE/HE/A2` 身份和单位无法不猜测地冻结。数据集级 evidence gap 不能 partial，保持 `BLOCKED/CHANNEL_IDENTITY_AND_UNIT_EVIDENCE_MISSING`，未提交。
- `ds005868`：48/48 continuous SET/FDT 为 500 Hz、60 Hz；唯一 32 路签名的 type/unit 全为 `n/a`，JSON 不声明 EEG/EOG/Misc 数量，SET `chanlocs.type` 为空，`LE/HE/A2` 身份和单位无法不猜测地冻结。数据集级 evidence gap 不能 partial，保持 `BLOCKED/CHANNEL_IDENTITY_AND_UNIT_EVIDENCE_MISSING`，未提交。
- `ds003751`：`BLOCKED`。31/40 FDT 确定性截断；剩余 9 条为 SET 129 EEG（含 E129 reference）但 JSON 128 EEG，且 boundary removed-sample intervals 需要当前 continuous adapter 尚未实现的 discontinuity-aware window 合同；accepted scope 0/40，不提交。
- `ds004504`：raw-only 88 条 embedded SET（另 88 条 derivatives 由冻结范围合同排除）；500 Hz、50 Hz、19 EEG `microV`。显式 fail-closed `NO_EVENTS` 合同通过 72/72 tests，冻结 `code-20261125-04`。已提交到 `tione_group`：task `train-1724989871473044224`，run `run-20261124T164650Z-ds004504`，候选 `r001`；当前 `PENDING/FIFOBlocked`。run：`cos://omni-eeg-processed-1442740494/datasets/ds004504/runs/run-20261124T164650Z-ds004504/`。
- `ds004771`：`DEATH_SENTENCE/CHANNEL_IDENTITY_UNIT+EPOCHED_SEMANTICS_UNRESOLVED`。61/61 SET/FDT 网格闭合，统一 256 Hz、60 Hz、34 通道；48,792 个事件时钟和 bounds 通过。但 2,074 个 channels.tsv 行全部 `type=n/a, units=n/a`，SET chanlocs type/unit 也为空，JSON 只有聚合 `EEGChannelCount=34`。整套包含 29,206 个 epoch，每 epoch 334 点（约 1.30 秒），当前没有获批的 trial-as-recording 合同，禁止拼成连续记录。0/61 可合法提交，未读取 FDT 波形、未提交 TIONE。
- `ds007081`：41/41 MATLAB v5 SET/FDT 的 header 与 FDT 字节网格完全匹配，均为 1000 Hz、50 Hz、32 个 EEG 通道和 `microV`；但每条均为 742–800 trials 的 epoched 数据，当前 continuous adapter 明确拒绝 `trials>1`，且 41/41 缺 exact `events.tsv` 与完整波形 QC，保持 `SOURCE_GATE/EVIDENCE_MISSING+ADAPTER_ERROR`，未提交。
- `ds004940`：48/48 为 BDF；40 条有 exact sidecar，但 `sub-004/007/012` Active 与 `sub-019` Passive 的 8 条 split run 只有未拆分的 task 级 channels/events，缺权威事件分区边界和逐 run 时钟；48/48 exact EEG JSON 也缺失。证据缺失不属于批准 partial reason，保持 `BLOCKED/RUN_EVENT_PARTITION_CLOCK_UNRESOLVED`，未提交。
- `ds004152`、`ds004264`：COS 源范围已标记完整且格式 census 识别为 BrainVision，但现有证据仅到 `FORMAT_ONLY`；recording 数与配对、metadata/工频、通道/单位、事件、完整波形 QC 和 adapter readiness 均未证明，保持 `SOURCE_GATE/EVIDENCE_MISSING`。
- `ds004284`：COS 源范围已标记完整，格式 census 为 BrainVision 加 `.gz`；现有证据仅到 `FORMAT_ONLY`，且压缩配套语义、recording 数与配对及全部后续门禁均未证明，保持 `SOURCE_GATE/EVIDENCE_MISSING`。
- `ds004477`：`DEATH_SENTENCE/CHANNEL_IDENTITY_CONFLICT`。9/9 BDF 的对象网格、2048 Hz、50 Hz 和 12,960 个事件时钟/bounds 均通过；但 TSV/JSON 将除 Status 外 79 行全部声明为 EEG，其中包含 `GSR1/GSR2`（`nS`）、`Temp`（`Celsius`）、Resp、Plet 等明确辅助生理信号。不得按名称猜测重分类，且身份/单位冲突不在 partial 白名单；0/9 可合法提交，未读取完整波形、未提交 TIONE。
- `ds004561`：COS 源范围已标记完整且 full format census 为 BrainVision 加 `.gz`，但压缩配套语义、recording 配对及全部 recording 级门禁未证明，保持 `SOURCE_GATE/EVIDENCE_MISSING`。
- `ds004572`：COS 源范围已标记完整，但 BrainVision format census 仅为 sampled；完整格式范围、recording 配对及全部 recording 级门禁未证明，保持 `SOURCE_GATE/EVIDENCE_MISSING`。
- `ds004603`：full census 覆盖 470 个对象，包含 37 组 BrainVision、61 个 derivative FIF 和 31 个 MAT；当前不能把所有扩展名直接当独立原始 recording，raw/derivative 权威范围及全部 recording 级门禁未闭合，保持 `SOURCE_GATE/EVIDENCE_MISSING+SCOPE_UNRESOLVED`。
- `ds004816`：已按 raw BrainVision scope 核对 20 条；20/20 缺精确 `channels.tsv`，其中 2/20 还缺精确 `events.tsv`，strict metadata gate 为 0/20，通过前不下载全波形、不提交，保持 `SOURCE_GATE/EVIDENCE_MISSING`。`ds004817` 同样核对 20 条，20/20 缺精确 `channels.tsv`、3/20 缺精确 `events.tsv`，保持 `SOURCE_GATE/EVIDENCE_MISSING`。
- `ds003690`：当前 375/375 精确 sidecar、500 Hz、50 Hz、通道/单位和 sample origin 0 metadata 已通过；但全部 signal 是约 35–71 MB 的 MATLAB v5 内嵌 SET，小对象审计无法证明逐 recording sample grid 与事件上界，开发机又禁止下载完整波形。旧 V1 的 100 Hz/逐窗 z-score/尾窗补零 publication 不能替代当前证据，保持 `SOURCE_GATE/RECIPE_MISMATCH+EVIDENCE_MISSING`，未提交。
- `ds004043`：`RUNNING`，complete 20/20；首 run 因冻结 contract JSON 字段名不匹配失败，源数据门禁未变。`code-20261125-74` 已用 reader 精确字段并通过 144/144 测试；TIONE `train-1725388295318658944`，`run-20261125T055823Z-ds004043`，candidate `r001`。
- `ds004951`：23/23 raw BrainVision sidecar 的 `PowerLineFrequency` 均未解决，strict metadata gate 为 0/23；不依据地理位置或频谱猜 50/60 Hz，不下载全波形、不提交，保持 `SOURCE_GATE/EVIDENCE_MISSING`。
- `ds004356`：当前权威 raw 根下 strict `_eeg.vhdr` 发现数为 0；历史 24 对 BrainVision census 未能映射为可提交的标准 BIDS signal scope，保持 `SOURCE_GATE/SCOPE_UNRESOLVED`。
- `ds004315`：full census 覆盖 493 个对象，包含 50 组 SET/FDT 与一个 code 目录 chanloc MAT；格式配对存在，但 metadata、工频、通道/单位、事件和完整波形 QC 未证明，保持 `SOURCE_GATE/EVIDENCE_MISSING`。
- `ds004582`、`ds004840`：旧 pretrain V1 publication 分别通过 73 条/12,367 行和 51 条/3,953 行，但均使用 100 Hz、逐窗 z-score、尾窗补零，与当前 V2 recipe 不一致；当前原始源与 recording 级门禁未证明，保持 `SOURCE_GATE/RECIPE_MISMATCH+EVIDENCE_MISSING`。
- `ds004706`：已重新从原始 BDF 闭合 298 条 header/grid/通道/单位/工频与事件门禁；精确隔离 15 条负 onset/sample recording，283 条 partial 已提交 `tione_group`，等待 worker 全波形 QC 和发布验收。
- `ds001787`、`ds002721`：旧 pretrain V1 publication 分别通过完整范围 40 条/9,958 行与 185 条/9,554 行，并记录远端训练读回通过；但 publication 本身没有当前 V2 recipe 证明，亦未携带完整原始 recording 门禁，因此保持 `SOURCE_GATE/RECIPE_EVIDENCE_MISSING+V2_GATE_MISSING`。
- `ds002680`：仅发现 `gonogo-image-erp-1s-native` 旧 release；2000 对象 sampled census 未定位任何信号扩展名，也没有 pretrain V1，完整 recording 源范围与全部门禁未知，保持 `SOURCE_GATE/SCOPE_UNRESOLVED+EVIDENCE_MISSING`。
- `ds003775`：旧 pretrain V1 publication 通过完整范围 153 条/3,672 行并记录训练读回通过，但 publication 不含当前 V2 recipe 证明，原始 recording 级门禁也未记录，保持 `SOURCE_GATE/RECIPE_EVIDENCE_MISSING+V2_GATE_MISSING`。
- `ds003825`：当前权威范围已闭合为 50 组 raw BrainVision；50/50 有 exact EEG JSON 和 events，但整个源前缀 0/50 有 `channels.tsv`，无法权威证明逐行 EEG/辅助通道身份与物理单位。该数据集级 evidence gap 不能用 partial 绕过，accepted 0/50，登记 `DEATH_SENTENCE/EVIDENCE_MISSING_CHANNEL_IDENTITY_AND_UNIT`；未下载完整波形、未提交 TIONE。证据：`reports/qc_issues/ds003825_missing_channel_contract_block_20261125.json`。
- `ds003885`：full census 为 24 组 raw BrainVision、24 组 derivative SET/FDT 和 161 个多为 derivatives 的 MAT；仅有 stimulus native 旧 release，当前 V2 范围及门禁未闭合，保持 `SOURCE_GATE/SCOPE_UNRESOLVED+EVIDENCE_MISSING`。
- `ds003887`：full census 为 24 组 raw BrainVision、24 组 derivative SET/FDT 和 154 个多为衍生分析的 MAT；仅有 stimulus native 旧 release，当前 V2 范围及门禁未闭合，保持 `SOURCE_GATE/SCOPE_UNRESOLVED+EVIDENCE_MISSING`。
- `ds004148`：旧 V1 仅为 partial scope，覆盖 896 条/26,880 行，且使用 100 Hz、逐窗 z-score、尾窗补零；仅前 2000 对象的 sampled census 已出现 900 个 SET 与 162 组 BrainVision，完整范围仍未知，保持 `SOURCE_GATE/PARTIAL_SCOPE+RECIPE_MISMATCH+EVIDENCE_MISSING`。
- `ds004356`：full census 有 24 组 BrainVision，但旧 release 都是 12 秒事件窗 native；完整 recording 的 metadata、工频、通道/单位、事件时钟、全波形 QC 和 V2 adapter 门禁未证明，保持 `SOURCE_GATE/EVIDENCE_MISSING`。
- `ds004408`：旧 V1 覆盖 380 条/7,607 行，但 recipe 为 100 Hz、逐窗 z-score、尾窗补零；当前 V2 的工频与其余原始 recording 门禁未证明，保持 `SOURCE_GATE/RECIPE_MISMATCH+EVIDENCE_MISSING`。
- `ds004902`：当前权威范围 218 条 continuous SET/FDT；213 条字节网格闭合，5 条 FDT 明确截断。排除这 5 条后仍有两条内部 EEGLAB boundary（sample 8,481/2,111），现有转换会跨真实时间断点；另有 19 条 SET/TSV 通道顺序冲突需 reviewed name binding。当前 213/218 partial 不合法，accepted 0/218，登记 `DEATH_SENTENCE/SOURCE_GRID+CHANNEL_ORDER+DISCONTINUITY_BLOCK`。证据：`reports/qc_issues/ds004902_source_grid_channel_boundary_block_20261125.json`。
- `ds004951`：full census 有 23 组 BrainVision，但仅有 letter-marker native；完整 recording 的 metadata、工频、通道/单位、事件时钟、全波形 QC 与 V2 adapter 门禁未证明，保持 `SOURCE_GATE/EVIDENCE_MISSING`。
- `ds004980`：17/17 BrainVision recording 已完成全波形解码且全部 finite/nonzero；约 500 Hz、64 EEG、`microvolts`，170,280 个事件全部在界内。17/17 EEG JSON 的 `PowerLineFrequency=n/a`，逐 recording 工频仍未解决；保持 `SOURCE_GATE/EVIDENCE_MISSING`，未提交。证据：`reports/preflight/ds004980_gate_summary.json`。
- `ds005095`：full census 有 48 组 BrainVision，但仅有 Sternberg marker native；逐 recording 工频、通道/单位、事件时钟、全波形 QC 和 V2 adapter 门禁未证明，保持 `SOURCE_GATE/EVIDENCE_MISSING`。
- `ds005121`：full census 有 39 组 BrainVision，但仅有 stimulus marker native；逐 recording 工频、通道/单位、事件时钟、全波形 QC 和 V2 adapter 门禁未证明，保持 `SOURCE_GATE/EVIDENCE_MISSING`。
- `ds005131`：63/63 native 行可读，500 Hz、每条 64 EEG、单位 V，232,204 个事件均带 sample/onset；但 63/63 acquisition power-line 未解决，且无完整 finite 波形 QC，保持 `SOURCE_GATE/EVIDENCE_MISSING`。
- `ds005178`：历史简表没有已验收 source prefix，也无匹配 Lance release；格式范围和 adapter 均未确定，waveform quickcheck 未执行，保持 `ADAPTER_BLOCKED/SOURCE_UNLOCATED+DIRECT_SOURCE_ADAPTER_NEEDED`。
- `ds005189`：旧 V1 通过完整范围 30 条/6,970 行，但使用 100 Hz、逐窗 z-score、尾窗补零；当前 V2 原始 recording 门禁未证明，保持 `SOURCE_GATE/RECIPE_MISMATCH+EVIDENCE_MISSING`。
- `ds005273`：full census 有 33 个 VHDR、仅 32 个 EEG 对象，存在明确配对计数差；旧 complete-recordings native 还是 partial，未匹配 header 与全部 V2 门禁未闭合，保持 `SOURCE_GATE/PAIRING_ERROR+PARTIAL_SCOPE+EVIDENCE_MISSING`。
- `ds005511`：仅前 2000 对象的 sampled census 已有 495 个 SET，完整范围仍未知；旧 complete-recordings native 为 partial，SET 是 embedded 还是外置 partner 及全部 V2 门禁未闭合，保持 `SOURCE_GATE/SAMPLED_SCOPE+PARTIAL_SCOPE+EVIDENCE_MISSING`。
- `ds005586`：23/23 BrainVision metadata 门禁闭合，但 task `train-1725301664732284672` / run `run-20261125T030617Z-ds005586` 在 `sub-015` 以 `PSD_POWER_LINE_UNRESOLVED` 失败，未生成 release。批次提升规则已更新为“一条可靠确认且已有证据无相反频率即可统一陷波”，不要求全批 PSD；但该失败 run 没有持久化任何可靠确认 evidence，因此尚不能合法提升或重试。
- `ds005594`：full census 有 16 组 BrainVision，但旧产物仅为 alphabetic-marker native；完整 recording 门禁未证明，保持 `SOURCE_GATE/EVIDENCE_MISSING`。
- `ds005620`：202/202 BrainVision metadata 门禁闭合；task `train-1725304562937628416` / run `run-20261125T031202Z-ds005620` 完成 1,212/1,212 文件、83.00 GB staging 后，在首条 `sub-1010/.../awake_acq-EC` 以 `PSD_POWER_LINE_UNRESOLVED` 失败，未生成 release，也未持久化可靠确认 evidence。按新规则不做全批 PSD 扫描，禁止无证据自动重试。
- `ds005697`：full census 有 51 个 SET，但 embedded/外置数据布局未识别，旧产物仅为 perceive-imagine native；metadata、工频、通道/单位、事件时钟、全波形 QC 和 adapter 门禁未证明，保持 `SOURCE_GATE/EVIDENCE_MISSING`。
- `ds003846`：旧 V1 通过完整范围 50 条/8,203 行，但使用 100 Hz、逐窗 z-score、尾窗补零；当前 V2 原始 recording 门禁未证明，保持 `SOURCE_GATE/RECIPE_MISMATCH+EVIDENCE_MISSING`。
- `alexmi`：别名 `moabb-alexmi` 的旧 V1 通过完整范围 8 条/402 行和远端训练读回，但 publication 无 recipe 结构，当前 V2 原始门禁未证明，保持 `SOURCE_GATE/RECIPE_EVIDENCE_MISSING+V2_GATE_MISSING`。
- `bnci2014-004`：旧 V1 acceptance/publication 通过完整范围并有 9,472 行，但本地证据没有当前 V2 recipe 或原始 recording 门禁，保持 `SOURCE_GATE/RECIPE_EVIDENCE_MISSING+V2_GATE_MISSING`。
- `bnci2014-008`、`bnci2014-009`、`bnci2015-001`：旧 V1 acceptance/publication 与远端读回均通过，分别有 1,088、600、6,024 行；但现有证据没有当前 V2 recipe 或原始 recording metadata/全波形门禁，均保持 `SOURCE_GATE/RECIPE_EVIDENCE_MISSING+V2_GATE_MISSING`。
- `bnci2015-003`、`bnci2015-004`：旧 V1 完整范围和远端读回通过，分别 346、4,959 行；源前缀首对象分别为 `s1.mat`、`A.mat`，但 MATLAB 内部 recording 结构、当前 V2 recipe 与原始门禁未证明，保持 `SOURCE_GATE/RECIPE_EVIDENCE_MISSING+V2_GATE_MISSING`。
- `cinebrain-eeg-hf`：别名 `cinebrain-eeg` 的旧 V1 有 13,478 行，但现有 source probe 只证明目录存在，未识别实际格式，也无逐数据集 preflight/全波形证据，保持 `SOURCE_GATE/RECIPE_EVIDENCE_MISSING+V2_GATE_MISSING`。
- `epflp300`：full census 覆盖 200 个对象，其中 192 个 MAT、8 个 ZIP；旧产物仅为 `event-eeg-1s-native` 事件窗范围，不能代表完整 recording。完整 recording 权威范围、MAT/ZIP 关系及全部 V2 门禁未证明，保持 `SOURCE_GATE/SCOPE_UNRESOLVED+EVIDENCE_MISSING`。
- `hms`：源端仅定位到一个 ZIP 容器；容器内部 recording 清单与实际信号格式未审计，旧 `partial-recordings-eeg-native` 明确为 partial。完整范围和全部 V2 门禁未证明，保持 `SOURCE_GATE/SCOPE_UNRESOLVED+PARTIAL_SCOPE+EVIDENCE_MISSING`。
- `weibo2014`：full census 覆盖 7 个对象，其中 4 个 MAT、3 个 ZIP；旧产物仅为 `event-eeg-4s-native` 事件窗范围，不能代表完整 recording。完整 recording 权威范围、MAT/ZIP 关系及全部 V2 门禁未证明，保持 `SOURCE_GATE/SCOPE_UNRESOLVED+EVIDENCE_MISSING`。
- `bbbd`：旧 native acceptance 覆盖 1,158 条 BDF recording；旧 V1 完整范围发布 33,697 行并通过远端读回，但使用旧 100 Hz channel-zscore recipe。当前 V2 recording 门禁未证明，保持 `SOURCE_GATE/RECIPE_MISMATCH+V2_GATE_MISSING`。
- `ds005262`：旧 V1 acceptance 声称完整范围 186 条/9,108 行，但其 native v001 缺 preflight、acceptance、publication、state 四类门禁文件，旧 V1 也无显式远端训练读回证据；当前 V2 recipe 与 recording 门禁未证明，保持 `SOURCE_GATE/NATIVE_RELEASE_EVIDENCE_MISSING+RECIPE_EVIDENCE_MISSING+V2_GATE_MISSING`。
- `tusz`：`DEATH_SENTENCE/CHANNEL_IDENTITY_UNRESOLVED`。当前已完成 7,364/7,364 EDF header census：57 种布局、17–128 个存储通道，采样率为 250/256/400/512/1000 Hz；辅助 EKG/EOG/EMG/RESP/PHOTIC/IBI/BURSTS/SUPPR 与头皮信号共存，且部分辅助标签也带 `EEG` 前缀。源端没有逐行权威 channel type 合同，不能按名称猜 EEG Lance 子集；0/7,364 可合法提交，未读取波形、未提交 TIONE。
- `tueg`：`DEATH_SENTENCE/CHANNEL_IDENTITY_AND_DUPLICATE_SCOPE_UNRESOLVED`。权威前缀含 69,672 条 EDF、约 1.760 TB、40+ layouts；辅助 EKG/LOC/ROC 等与头皮信号同样使用 `EEG` 前缀和 `uV`，无法由 label 权威分类。源描述还明确称该目录是 residual duplicate，main TUH corpora 已迁移，净新增范围未闭合。不能靠 partial 解决逐通道身份或整套去重范围，未读取完整波形、未提交 TIONE。
- `ds005514`：`PUBLISHED` complete 2,885/2,885，`r001`；Lance shape `[channels,2000]`，尾窗 `[channels,<2000]`；77,321 窗、2,885 尾窗、0 quarantine。权威 TIONE `train-1725494560597083904` / `run-20261125T092927Z-ds005514`；acceptance、publication、首/中/尾正式读回、Lance 与 `_SUCCESS` 一致。最终路径 `cos://omni-eeg-processed-1442740494/datasets/ds005514/releases/pretrain_V2/r001/data.lance/`。372 条冲突 MI-BCI sourcedata BDF 按 `BIDS_PRIMARY_EXCLUDE_SOURCEDATA` 排除。
- `egobrain`：旧 native acceptance 有 40 行，旧 V1 完整范围发布 22,805 行但无显式远端训练读回；旧 100 Hz channel-zscore recipe 与当前 V2 不一致，原始 recording 门禁未证明，保持 `SOURCE_GATE/RECIPE_MISMATCH+V2_GATE_MISSING`。
- `inria-large`：通过别名 `inria` 匹配旧 native 694 行和旧 V1 完整范围 26,601 行，但无显式远端训练读回；源端为单一 ZIP，当前 V2 recording 门禁未证明，保持 `SOURCE_GATE/RECIPE_MISMATCH+V2_GATE_MISSING`。
- `mind-to-play`：旧 native acceptance 覆盖完整范围 73 行，旧 V1 完整范围发布 8,362 行但无显式远端训练读回；旧 100 Hz recipe 与当前 V2 不一致，原始 recording 门禁未证明，保持 `SOURCE_GATE/RECIPE_MISMATCH+V2_GATE_MISSING`。
- `nmt`：旧 native acceptance 覆盖完整范围 2,417 行，旧 V1 完整范围发布 177,096 行但无显式远端训练读回；direct source 为单一 ZIP，当前 V2 recording 门禁未证明，保持 `SOURCE_GATE/RECIPE_MISMATCH+V2_GATE_MISSING`。
- `nmt-events`：旧 native acceptance 覆盖完整范围 1,075 行，旧 V1 完整范围发布 140,425 行但无显式远端训练读回；direct source 为单一 ZIP，当前 V2 recording 门禁未证明，保持 `SOURCE_GATE/RECIPE_MISMATCH+V2_GATE_MISSING`。
- `spanishbcbl`：旧 native acceptance 覆盖完整范围 117 行，旧 V1 完整范围发布 10,022 行但无显式远端训练读回；旧 100 Hz recipe 与当前 V2 不一致，原始 recording 门禁未证明，保持 `SOURCE_GATE/RECIPE_MISMATCH+V2_GATE_MISSING`。
- `sparrkulee`：旧 native acceptance 覆盖完整范围 718 行，旧 V1 完整范围发布 60,752 行且 publication 记录远端训练读回通过；但旧 100 Hz recipe 与当前 V2 不一致，原始 recording 门禁未证明，保持 `SOURCE_GATE/RECIPE_MISMATCH+V2_GATE_MISSING`。
- `wearbci`：旧 native acceptance 覆盖完整范围 560 行，旧 V1 完整范围发布 6,075 行但无显式远端训练读回；旧 100 Hz recipe 与当前 V2 不一致，原始 recording 门禁未证明，保持 `SOURCE_GATE/RECIPE_MISMATCH+V2_GATE_MISSING`。
- `ds005811`：旧 native acceptance 覆盖完整范围 448 条 SET recording，旧 V1 完整范围发布 8,821 行但无显式远端训练读回；旧 100 Hz recipe 与当前 V2 不一致，原始 recording 门禁未证明，保持 `SOURCE_GATE/RECIPE_MISMATCH+V2_GATE_MISSING`。
- `ease-tsd`：旧 native acceptance 完整遍历 1,057 行、零 quarantine，旧 V1 完整范围发布 16,074 行但无显式远端训练读回；当前 V2 recording 门禁未证明，保持 `SOURCE_GATE/RECIPE_MISMATCH+V2_GATE_MISSING`。
- `ds004395`：`DEATH_SENTENCE/CHANNEL_TYPE_UNIT_CONFLICT`。旧“源定位器过期”结论已被实时复核推翻：权威源可在 `omni-eeg-1442740494/omni-eeg-01/reve_raw/openneuro/ds004395/` 访问，共 6,483 条（4,978 EDF + 1,505 BDF，约 9.58 TB）。全量门禁发现 3 条非法 header、2 条二进制网格不齐、294 条事件 onset 越界；这些可确定性隔离，但另有 13 条 BDF 把 `nS`/`Celsius`/`n/a` 传感器行明确标为 EEG，属于非 partial 白名单的身份/单位冲突。当前 0/6,483 可合法提交，未读取波形、未提交 TIONE。
- `ds002218`：18/18 SET 均有 exact sidecar，但通道身份、单位、工频与事件 sample origin 全部未解决；没有完整波形解码，保持 `SOURCE_GATE/METADATA_ERROR+EVIDENCE_MISSING`。
- `ds004362`：1,526 条 SET 的通道身份已明确；历史 preflight 发现 36 条 EDF grid/通道数异常和 12 条不支持的 WFDB annotation，且无全波形 QC，保持 `SOURCE_GATE/DETERMINISTIC_SOURCE_ERROR+ADAPTER_ERROR+EVIDENCE_MISSING`。
- `ds004841`：147 条 SET 的通道身份明确，但 136 条存在非有限或负 event onset，历史 preflight 另有事件越界；无全波形 QC，保持 `SOURCE_GATE/METADATA_ERROR+EVIDENCE_MISSING`。
- `nki-naturalistic-viewing`：raw 区 942 个 signal entry 严格组成 471 个 SET/VHDR 同 stem 对；但 471/471 VHDR 声明的 EEG 数据伙伴全部不存在，实际波形仅在 471 个 SET（COS HEAD 合计 258.5 GB）。259,504 个 BIDS events 的 onset/sample 不存在统一 origin，且与原生 VMRK 不能逐行绑定；保持 `ADAPTER_BLOCKED/MISSING_BRAINVISION_DATA_PARTNERS+EVENT_CLOCK_UNRESOLVED`，未提交。
- `jeong2020`：213/213 BDF 对象 grid、1,000 Hz/60 Hz 和 39,051 个事件边界全部闭合；但全部 sidecar/JSON 将 71 路声明为 EEG，而每条名称明确包含 4 EOG 与 7 EMG（共 2,343 个辅助行被标 EEG）。保持 `SOURCE_GATE/CHANNEL_IDENTITY_CONFLICT`，未提交。
- `toonfaces`：26/26 EDF 对象 grid、500 Hz 和 23,678 个事件边界全部闭合；header/TSV 数量差仅来自标准 `EDF Annotations` 通道，38 路信号顺序一致。26/26 工频缺可靠 50/60 Hz 证据，保持 `SOURCE_GATE/POWER_LINE_FREQUENCY_UNRESOLVED`，未提交。
- `inner-speech-nieto`：30/30 raw BDF 全部缺 EEG JSON、channels.tsv 与 events.tsv；README 的 136-channel/5,640-trial 总体描述不足以证明逐通道身份、单位、工频和内嵌事件时钟，90 FIF 为处理产物。保持 `SOURCE_GATE/METADATA_MISSING+EVENT_CLOCK_UNRESOLVED`，未提交。
- `ds006437`：正式 `PUBLISHED r001`，complete 63/63，6,084 窗、61 个变长尾窗，shape `[64,2000]` / tail `[64,<2000]`。权威 task `train-1725421527077357312` / run `run-20261125T070423Z-ds006437` 终态 `SUCCEED`；63/63 完整波形 QC、0 issues，acceptance/publication passed，远程总行数 6,084，训练消费者首/中/末索引 `[0,3042,6083]` 实读 3 行，Lance 内容与最后写入的 `_SUCCESS` 一致。正式 Lance：cos://omni-eeg-processed-1442740494/datasets/ds006437/releases/pretrain_V2/r001/data.lance/。

## 本轮失败但保留的 run

| dataset | run | 结果 | 证据位置 |
|---|---|---|---|
| ds002722 | PUBLISHED | complete | `[channels,2000]; tail [channels,<2000]` | 8281 | `qc-v022-20261124T075946Z` | cos://omni-eeg-processed-1442740494/datasets/ds002722/releases/pretrain_V2/v022/data.lance/ |
| ds007648 | PUBLISHED_PARTIAL | partial | `[channels,2000]; tail [channels,<2000]` | 5598 | `qc-v023-20261124T075949Z` | cos://omni-eeg-processed-1442740494/datasets/ds007648/releases/pretrain_V2_partial/v023/data.lance/ |
| ds003969 | PUBLISHED | complete 392/392 | `[channels,2000]; tail [channels,<2000]` | 24135 | `run-20261125T091208Z-ds003969` | cos://omni-eeg-processed-1442740494/datasets/ds003969/releases/pretrain_V2/r001/data.lance/ |
| ds004015 | PUBLISHED | complete | `[channels,2000]; tail [channels,<2000]` | 17040 | `run-20261124T121214Z-ds004015` | cos://omni-eeg-processed-1442740494/datasets/ds004015/releases/pretrain_V2/r002/data.lance/ |

## 当前云端任务

| dataset | 状态 | 范围 | Lance 形状 | 窗口数 | 当前权威 run | 任务 / 下一步 |
|---|---|---|---|---:|---|---|
| ds004706 | RUNNING | partial 283/298 | `[channels,2000]; tail [channels,<2000]` | 待验收 | `run-20261125T021920Z-ds004706` | TIONE `train-1725278025886598912`；RUNNING |
| ds005507 | RUNNING | complete candidate 1812/1812 | `[channels,2000]; tail [channels,<2000]` | 待验收 | `run-20261125T094520Z-ds005507` | TIONE `train-1725502570758528896`；RUNNING |
| ds005508 | RUNNING | complete candidate 3342/3342 | `[channels,2000]; tail [channels,<2000]` | 待验收 | `run-20261125T071015Z-ds005508` | TIONE `train-1725424477893156736`；RUNNING |
| ds005509 | RUNNING | complete candidate 3326/3326 | `[channels,2000]; tail [channels,<2000]` | 待验收 | `run-20261125T071350Z-ds005509` | TIONE `train-1725426295461240704`；RUNNING |
| ds005511 | RUNNING | complete candidate 3100/3100 | `[channels,2000]; tail [channels,<2000]` | 待验收 | `run-20261125T112035Z-ds005511` | TIONE `train-1725550498994839424`；RUNNING |
## 维护规则

- 每个数据集在主表只保留一行当前权威状态；失败和被替代的 run 放入历史表，不冒充已处理。
- `PUBLISHED` 必须同时有实际 Lance 路径、窗口数、recording 范围、shape、acceptance、training readback 和 `_SUCCESS` 证据；`PUBLISHED_LEGACY` 明确标注旧合同没有 `_SUCCESS`，但其余证据已通过。
- 完整窗固定为 200 Hz × 10 秒 = 2000 点；尾窗不补零，因此第二维可以小于 2000。
- 新任务提交后写 `RUNNING`；终态后立即更新为 `PUBLISHED` 或 `FAILED`。
- 详细问题写 `docs/质检问题记录.md`；机器可读错误写 `runs/<dataset>/<run_id>/qc/issues.jsonl`。
- 不能仅凭 TIONE `SUCCEED` 或存在目录就标为 `PUBLISHED`。
- 后续新合同使用每数据集独立的 `r001`、`r002`……作为正式 release 编号。失败和重试只新增 `run-*`，不递增 release；只有全部发布门禁通过后才占用当前 `rNNN`。历史 `vNNN` 路径保持不变。
