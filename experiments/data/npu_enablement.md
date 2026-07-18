# NPU 打通攻坚实录（nubia P0210 / canoe，Android 16，2026-07-18）

目标：在 Qualcomm 手机上跑通 LiteRT-LM 的 `--backend=npu`。

## 组件清单与来源（三层全部就位并验证）

1. **QNN 运行时（HTP 库）**：设备 `/vendor/lib64/` 自带 `libQnnHtp.so`、`libQnnHtpV81.so`、
   `libQnnHtpV81Stub.so`、`libQnnSystem.so`（HTP V81）。
   注意：vendor 库 shell 不可读（SELinux），且不在 `public.libraries.txt`（仅 adsprpc/cdsprpc/OpenCL 等公开）。
2. **Dispatch 桥 `libLiteRtDispatch_Qualcomm.so`**：源码自编译
   `bazel build --config=android_arm64 @litert//litert/vendors/qualcomm/dispatch:dispatch_api_so`
   （与 litert_lm_advanced_main 同一 LiteRT pin，规避上游 issue #6889 的 API 错位）。
3. **NPU 打包模型**：`gemma-4-E2B-it_qualcomm_sm8750.litertlm`（2.8 GB，litert-community 公开）。
   内含 `tf_lite_aux` / `tf_lite_embedder` / `tf_lite_per_layer_embedder` 段（strings 验证），
   即第 8 章所述的多编译子模型打包。
4. **QAIRT 库（宿主+DSP 两侧）**：无需高通账号——LiteRT-LM WORKSPACE 的 `qairt()` 规则
   已在 bazel 外部仓库缓存备好完整 SDK（`external/qairt/`），含
   `lib/aarch64-android/`（宿主 libQnnHtp.so/Prepare/System/V81Stub）与
   `lib-safe/hexagon-v81/unsigned/`（DSP 侧 V81 skel/system）。

## 进展到执行的完整链路（全部打通）

`--backend=npu --litert_dispatch_lib_dir=<qnn dir>`（flag 见 shared_flags.cc:129）：
dispatch 桥加载 → QNN manager 初始化 → libQnnSystem/libQnnHtp 加载 →
模型多段读取（TF_LITE_AUX/EMBEDDER）→ DISPATCH_OP 解析为 DispatchDelegate →
QNN context 创建成功（嵌入模型 bytecode 1.18 GB + aux 430 KB）。

## 两道墙（都已定性）

**墙一（决定性）：生产 ROM 拒绝未签名 DSP skel。**
`qnn-platform-validator --backend dsp --testBackend`：
calculator 测试执行失败，"Please use testsig if using unsigned images"。
设备为 user/release-keys 生产签名，第三方未签名 skel 无法上 DSP 执行。
（vendor 内的签名 skel 又因 SELinux 不可读、不在 public 命名空间。）
→ stock ROM 下 adb 路线无法执行 HTP 工作负载；工程机/root/厂商渠道才行。

**墙二（在墙一之后）：预编译 context 与 HTP 架构绑定。**
E2B 包的内嵌 context binary 面向 sm8750（Snapdragon 8 Elite，V79 代）预编译；
本机为更新平台（canoe/soc_id 660，HTP V81）。预编译 HTP 二进制按架构绑定，
v0.13.1 的 dispatch 路径无 JIT 兜底——SoC 不匹配即无法执行。
这正是 Google/Qualcomm 按 SoC 分别发布模型包（文件名即带 sm8750）的根因。

## 结论与未试路径

NPU 的「封闭」至此有了完整的机制级证据：签名绑定（ROM 层）+ 架构绑定（模型包层）。
未试：Qualcomm AI Hub 寻找匹配本机 SoC 的包（需账号，新平台可能尚未发布）、
工程机/已 root 设备、sm8750 真机（其 ROM 据报可跑，如 S25 Ultra 案例）。

---

## 附：第二台设备（HONOR MEP-AN00，canoe→V79，2026-07-18 补测）

换机重试，设备画像完全不同：
- **SoC 为 V79 代**（/odm/lib64 有 libQnnHtpV79Stub/Skel）——与 E2B 包的目标 sm8750 **匹配**，第一台的架构绑定墙不存在。
- **/odm 的 QNN 库 shell 可读**（libQnnHtp/System/V79Stub/V79Skel 均可 cp）——可用设备原厂签名 stub，而非未签名版。

实际进展与卡点：
1. `qnn-platform-validator --backend dsp --testBackend`：calculator 执行仍失败
   "Please use testsig if using unsigned images"——与 nubia 同类的 ROM 级 DSP 限制
   （换用 /odm 原厂 skel 原位路径亦同）。
2. LiteRT-LM 链路：先解两个版本错配——/odm 的 libQnnSystem 1.4.0（需 ≥1.10）、
   libQnnHtp 2.27.0（需 ≥2.35），换 QAIRT 2.46 宿主库后通过；
   随后卡在 QNN manager 建 backend/device 一步（DSP 握手层），与 validator 同墙。
3. 决定性对照：再换 **QAIRT 2.42 全套 V79 一致栈**（免登录公开下载，含 hexagon-v79 skel/stub）重测，validator 报同一错误——版本错位假设被排除，失败原因就是 ROM 的 DSP 访问策略。

小结：第二台证明"封闭"还有第三层——OEM ROM 的 DSP 访问策略与固件版本差。
同一模型包、同一工具链，在两家 OEM 的 ROM 上倒下的位置都不一样。
