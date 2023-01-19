python train_artvgg.py ^
--content_path "D:\MyProject\Dataset\coco2017\train2017" ^
--style_path "D:\nick\Desktop\jupyter notebook\styleTransfer\data\ukiyoe2photo" ^
--name artvgg_test ^
--model artvgg ^
--dataset_mode classify ^
--no_dropout ^
--load_size 256 ^
--crop_size 224 ^
--gpu_ids 0 ^
--batch_size 2 ^
--n_epochs 12 ^
--n_epochs_decay 3 ^
--display_freq 1 ^
--display_port 8097 ^
--display_env ArtVGG

