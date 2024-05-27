Our dataset is available [here](https://drive.google.com/drive/folders/16YihOAL6sIytNCKtrV7rU8SXAv8hmJVH?usp=drive_link).

We sample around 4k/0.5k/1k examples for train/valid/test splits on `ASQA` and approximately 4k/1k/1k examples on `QAMPARI` and `ELI5`. They are used directly in the *separate* setting. In the *combined* setting, we further sample around 1k/334/1k instances from the aforementioned examples to get 2,992/1,002/3,000 instances for joint training and inference.

On `EXPERTQA`, we remove 8 exmples that do not have human-revised answers since we need revised answers to derive sub-claims for correctness recall computation. This gives us the remaining 2,169 exmples. Given that certain groups of instances in EXPERTQA are evaluated in a closed-book setting or use a different retrieval mechanism, We manually retrieve the top 5 passages from Sphere (Piktus et al., 2021) again for each example to ensure consistency with other datasets. We use EXPERTQA for testing only.

See Appendix B.1 in our paper for more details.