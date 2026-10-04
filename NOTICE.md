# Attribution

SCPT is built on [TCP](https://github.com/htyao89/Textual-based_Class-aware_prompt_tuning),
[CoOp](https://github.com/KaiyangZhou/CoOp),
[Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch), and
[OpenAI CLIP](https://github.com/openai/CLIP).

- `trainers/clip_text/model.py` and `simple_tokenizer.py`, and the BPE vocabulary,
  derive from CLIP. The class-aware text blocks derive from TCP's modified text
  encoder. Upstream copyright: OpenAI, 2021; MIT license.
- The prompt learner, trainer organization, and dataset conventions derive from
  TCP/CoOp. TCP's repository carries the MIT notice for Kaiyang Zhou, 2021.
- Dassl is an external dependency, not vendored here.
- SCPT-specific changes implement relational binary prompts and semantic
  condensation. The release removes other method trainers and experiment suites.

The corresponding MIT copyright and permission notices are preserved in
`LICENSE`. Original upstream license files:
[TCP](https://github.com/htyao89/Textual-based_Class-aware_prompt_tuning/blob/main/LICENSE),
[CLIP](https://github.com/openai/CLIP/blob/main/LICENSE).

The code license does not grant rights to third-party dataset images or pretrained
weights. Obtain these from their respective providers and follow their terms.
Datasets, split manifests, pretrained weights, and experiment results are not
included in this repository.
