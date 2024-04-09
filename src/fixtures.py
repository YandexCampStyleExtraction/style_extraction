from peft import LoraConfig, IA3Config, AdaLoraConfig, PromptTuningConfig

from models.losses import ContrastiveLoss, TripletMarginWithDistanceLoss

AVAILABLE_PEFT = {
    'lora': LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias='none',
        target_modules=['query', 'value'],
    ),
    'adalora': AdaLoraConfig(
        target_r=8,
        init_r=12,
        lora_alpha=32,
        lora_dropout=0.1,
        bias='none',
        target_modules=['query', 'query'],
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
    'contrastive': ContrastiveLoss,
    'triplet': TripletMarginWithDistanceLoss
}

AVAILABLE_CLS_LOSSES = {
    'arcface',
    'sphereface',
    'cosface',
}
