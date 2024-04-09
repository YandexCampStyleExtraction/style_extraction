from peft import LoraConfig, IA3Config, AdaLoraConfig, PromptTuningConfig

from models.losses import ContrastiveLoss, TripletMarginLoss

AVAILABLE_PEFT = {
    'lora': LoraConfig(
        r=14,
        lora_alpha=32,
        lora_dropout=0.1,
        bias='none',
        target_modules=['query', 'value'],
    ),
    'adalora': AdaLoraConfig(
        target_r=14,
        init_r=20,
        lora_alpha=32,
        lora_dropout=0.1,
        bias='none',
        target_modules=['query', 'value'],
    ),
    'ia3': IA3Config(
        target_modules=['query', 'value', 'output'],
        feedforward_modules=['output'],
        # task_type=TaskType.SEQ_CLS
    ),
    'prompt': PromptTuningConfig(
        num_virtual_tokens=20,
        task_type="FEATURE_EXTRACTION",
    )
}

AVAILABLE_SSL_LOSSES = {
    # 'contrastive': ContrastiveLoss,
    'triplet': TripletMarginLoss
}

AVAILABLE_CLS_LOSSES = {
    'arcface',
    'sphereface',
    'cosface',
}
