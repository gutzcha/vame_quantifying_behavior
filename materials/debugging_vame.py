import vame

config = r"D:\Project- Electro\VAME\materials\working_dir\Unsupervised Learning Tutorial with VAME-May28-2024\config.yaml"

# OPTIONAL: Create behavioural hierarchies via community detection
# vame.community(config, show_umap=False, cut_tree=2)

vame.community_videos(config,videoType='.avi')