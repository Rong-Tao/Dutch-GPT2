# Note

This code was originally designed for use with RoBERTa models. However, during development, I encountered issues with kernel failures on specific Chinese-customized AMD GPUs. These failures were insanely difficult to diagnose and troubleshoot, leading to considerable challenges in project progress (I almost gave up).

Thanks to insights from a colleague, it was identified that the incompatibilities were related to the RoBERTa implementation when used with these specific GPUs. We recommend caution and additional testing if using this configuration. The issue seems to be specific to this GPU model and should not impact other hardware.
