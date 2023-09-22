---

quantization_modifiers:
  - !QuantizationModifier
    ignore: ["lm_head", "Embedding", "OPTLearnedPositionalEmbedding", "QuantizableBatchMatMul", "BMMLeftInput_QK", "BMMRightInput_QK", "BMMOutput_QK", "BMMLeftInput_PV", "BMMRightInput_PV", "BMMOutput_PV"]
    scheme_overrides:
      ReLU:
        input_activations: null
        output_activations: null
      LayerNorm:
        input_activations: null
        output_activations: null

---
