# Synthetic AE-explainability sweeps

Ground-truth recovery of a known relation through the project AE. `span_mean_r2` is the E1 per-factor (coordinate-level) recovery; `canon_min` is the E1 Bai-Ng *linear* (space-level) recovery; `kcca_min` is the E2 kernel-CCA *nonlinear* (space-level) recovery -- `kcca_min` >> `canon_min` means the f->g relation is real but nonlinear, so the linear probe understates it (space-level nonlinearity premium); `subspace_min` is decoder-span vs true-loading-span overlap.

## A. Bottleneck rank (linear DGP, true_rank=3, snr=6)

| encoder   |   n_factors | relation   |   true_rank |   snr |   recon_r2 |   span_mean_r2 |   span_max_r2 |   canon_min |   canon_mean |   kcca_min |   kcca_mean |   subspace_min |   n_active |   nonlin_premium |
|:----------|------------:|:-----------|------------:|------:|-----------:|---------------:|--------------:|------------:|-------------:|-----------:|------------:|---------------:|-----------:|-----------------:|
| relu      |           2 | linear     |           3 |     6 |      0.617 |          0.937 |         0.94  |       0.962 |        0.966 |      0.897 |       0.923 |          0.952 |          2 |              nan |
| relu      |           3 | linear     |           3 |     6 |      0.83  |          0.925 |         0.934 |       0.949 |        0.964 |      0.933 |       0.945 |          0.929 |          3 |              nan |
| relu      |           4 | linear     |           3 |     6 |      0.855 |          0.8   |         0.922 |       0.952 |        0.967 |      0.941 |       0.952 |          0.949 |          4 |              nan |
| relu      |           6 | linear     |           3 |     6 |      0.89  |          0.664 |         0.781 |       0.951 |        0.968 |      0.947 |       0.956 |          0.944 |          6 |              nan |
| linear    |           2 | linear     |           3 |     6 |      0.635 |          0.949 |         0.952 |       0.969 |        0.974 |      0.917 |       0.932 |          0.952 |          2 |              nan |
| linear    |           3 | linear     |           3 |     6 |      0.843 |          0.942 |         0.949 |       0.953 |        0.967 |      0.937 |       0.946 |          0.936 |          3 |              nan |
| linear    |           4 | linear     |           3 |     6 |      0.867 |          0.898 |         0.942 |       0.952 |        0.968 |      0.944 |       0.953 |          0.937 |          4 |              nan |
| linear    |           6 | linear     |           3 |     6 |      0.921 |          0.851 |         0.884 |       0.954 |        0.969 |      0.949 |       0.958 |          0.942 |          6 |              nan |
| pca       |           2 | linear     |           3 |     6 |      0.635 |          0.95  |         0.953 |       0.969 |        0.974 |      0.917 |       0.932 |          0.953 |          2 |              nan |
| pca       |           3 | linear     |           3 |     6 |      0.843 |          0.936 |         0.953 |       0.953 |        0.968 |      0.936 |       0.948 |          0.935 |          3 |              nan |
| pca       |           4 | linear     |           3 |     6 |      0.876 |          0.703 |         0.953 |       0.953 |        0.968 |      0.939 |       0.952 |          0.936 |          4 |              nan |
| pca       |           6 | linear     |           3 |     6 |      0.925 |          0.47  |         0.953 |       0.954 |        0.97  |      0.941 |       0.953 |          0.941 |          6 |              nan |

## B. Encoder vs noise (K=true_rank=3)

| encoder   |   n_factors | relation   |   true_rank |   snr |   recon_r2 |   span_mean_r2 |   span_max_r2 |   canon_min |   canon_mean |   kcca_min |   kcca_mean |   subspace_min |   n_active |   nonlin_premium |
|:----------|------------:|:-----------|------------:|------:|-----------:|---------------:|--------------:|------------:|-------------:|-----------:|------------:|---------------:|-----------:|-----------------:|
| relu      |           3 | linear     |           3 |     1 |      0.559 |          0.726 |         0.743 |       0.8   |        0.86  |      0.823 |       0.854 |          0.927 |          3 |              nan |
| linear    |           3 | linear     |           3 |     1 |      0.571 |          0.775 |         0.793 |       0.805 |        0.864 |      0.833 |       0.861 |          0.946 |          3 |              nan |
| pca       |           3 | linear     |           3 |     1 |      0.572 |          0.75  |         0.837 |       0.808 |        0.865 |      0.829 |       0.86  |          0.95  |          3 |              nan |
| relu      |           3 | linear     |           3 |     2 |      0.673 |          0.831 |         0.844 |       0.88  |        0.916 |      0.878 |       0.901 |          0.932 |          3 |              nan |
| linear    |           3 | linear     |           3 |     2 |      0.685 |          0.864 |         0.874 |       0.883 |        0.919 |      0.888 |       0.906 |          0.944 |          3 |              nan |
| pca       |           3 | linear     |           3 |     2 |      0.686 |          0.846 |         0.901 |       0.883 |        0.919 |      0.884 |       0.905 |          0.944 |          3 |              nan |
| relu      |           3 | linear     |           3 |     4 |      0.779 |          0.898 |         0.909 |       0.93  |        0.95  |      0.916 |       0.932 |          0.931 |          3 |              nan |
| linear    |           3 | linear     |           3 |     4 |      0.791 |          0.92  |         0.929 |       0.933 |        0.954 |      0.923 |       0.935 |          0.939 |          3 |              nan |
| pca       |           3 | linear     |           3 |     4 |      0.791 |          0.91  |         0.938 |       0.933 |        0.954 |      0.92  |       0.935 |          0.938 |          3 |              nan |
| relu      |           3 | linear     |           3 |     8 |      0.861 |          0.94  |         0.947 |       0.96  |        0.971 |      0.942 |       0.953 |          0.928 |          3 |              nan |
| linear    |           3 | linear     |           3 |     8 |      0.874 |          0.954 |         0.96  |       0.964 |        0.975 |      0.945 |       0.953 |          0.935 |          3 |              nan |
| pca       |           3 | linear     |           3 |     8 |      0.874 |          0.95  |         0.961 |       0.963 |        0.975 |      0.945 |       0.955 |          0.934 |          3 |              nan |
| relu      |           3 | linear     |           3 |    16 |      0.916 |          0.964 |         0.968 |       0.976 |        0.982 |      0.96  |       0.966 |          0.926 |          3 |              nan |
| linear    |           3 | linear     |           3 |    16 |      0.929 |          0.974 |         0.978 |       0.981 |        0.987 |      0.96  |       0.965 |          0.932 |          3 |              nan |
| pca       |           3 | linear     |           3 |    16 |      0.929 |          0.974 |         0.98  |       0.981 |        0.987 |      0.962 |       0.968 |          0.931 |          3 |              nan |

## C. Linear vs nonlinear relation (K=4, E2 on)

| encoder   |   n_factors | relation    |   true_rank |   snr |   recon_r2 |   span_mean_r2 |   span_max_r2 |   canon_min |   canon_mean |   kcca_min |   kcca_mean |   subspace_min |   n_active |   nonlin_premium |
|:----------|------------:|:------------|------------:|------:|-----------:|---------------:|--------------:|------------:|-------------:|-----------:|------------:|---------------:|-----------:|-----------------:|
| relu      |           4 | linear      |           3 |     6 |      0.855 |          0.8   |         0.922 |       0.952 |        0.967 |      0.941 |       0.952 |          0.949 |          4 |           -0.096 |
| linear    |           4 | linear      |           3 |     6 |      0.867 |          0.898 |         0.942 |       0.952 |        0.968 |      0.944 |       0.953 |          0.937 |          4 |           -0.112 |
| pca       |           4 | linear      |           3 |     6 |      0.876 |          0.703 |         0.953 |       0.953 |        0.968 |      0.939 |       0.952 |          0.936 |          4 |           -0.124 |
| relu      |           4 | interaction |           3 |     6 |      0.854 |          0.683 |         0.88  |       0.931 |        0.956 |      0.945 |       0.953 |        nan     |          4 |            0.028 |
| linear    |           4 | interaction |           3 |     6 |      0.882 |          0.815 |         0.924 |       0.957 |        0.968 |      0.946 |       0.954 |        nan     |          4 |           -0.06  |
| pca       |           4 | interaction |           3 |     6 |      0.884 |          0.702 |         0.96  |       0.956 |        0.968 |      0.947 |       0.954 |        nan     |          4 |            0.047 |
| relu      |           4 | tanh        |           3 |     6 |      0.86  |          0.695 |         0.804 |       0.887 |        0.896 |      0.946 |       0.953 |        nan     |          4 |            0.162 |
| linear    |           4 | tanh        |           3 |     6 |      0.867 |          0.772 |         0.814 |       0.883 |        0.896 |      0.946 |       0.953 |        nan     |          4 |            0.178 |
| pca       |           4 | tanh        |           3 |     6 |      0.877 |          0.603 |         0.828 |       0.883 |        0.896 |      0.945 |       0.952 |        nan     |          4 |            0.105 |