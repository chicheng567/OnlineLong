# Copyright 2024 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Simplified imports without lazy loading
from .configuration_qwen2 import Qwen2Config
from .tokenization_qwen2 import Qwen2Tokenizer

try:
    from .tokenization_qwen2_fast import Qwen2TokenizerFast
except ImportError:
    pass

try:
    from .modeling_qwen2 import (
        Qwen2ForCausalLM,
        Qwen2ForQuestionAnswering,
        Qwen2Model,
        Qwen2PreTrainedModel,
        Qwen2ForSequenceClassification,
        Qwen2ForTokenClassification,
    )
except ImportError:
    pass

__all__ = [
    "Qwen2Config",
    "Qwen2Tokenizer",
    "Qwen2TokenizerFast",
    "Qwen2ForCausalLM",
    "Qwen2Model",
    "Qwen2PreTrainedModel",
    "Qwen2ForQuestionAnswering",
    "Qwen2ForSequenceClassification",
    "Qwen2ForTokenClassification",
]
