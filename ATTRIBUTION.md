# Attribution and source provenance

This fork packages the Informer reference implementation by Haoyi Zhou and
collaborators, with the upstream README crediting Jieqi Peng for repository
implementation. Preserve the original acknowledgments and citation.

- Paper: Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li,
  Hui Xiong, and Wancai Zhang. *Informer: Beyond Efficient Transformer for Long
  Sequence Time-Series Forecasting*. AAAI 2021.
  <https://arxiv.org/abs/2012.07436>
- Upstream: <https://github.com/zhouhaoyi/Informer2020>
- Maintained fork: <https://github.com/EhsanAramide/Informer>
- Reference revision: `29f2a739226a509202a092b464163da81fa74960`.

The root `LICENSE` contains Apache License 2.0 text, while the retained upstream
README (now `README.upstream.md`) carries a CC BY-NC-SA badge. This discrepancy is recorded, not resolved
by this packaging work. The original license and README are preserved; no new
licensing terms or publication authorization are inferred. Publication is outside
this journey and should not rely on treating the discrepancy as resolved.

The original model and utility files remain unchanged. Maintained copies under
`src/informer` use namespaced imports, API documentation, and the utility
compatibility replacements recorded in the development/validation documents.
