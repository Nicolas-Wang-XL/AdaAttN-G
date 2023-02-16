python train_artvgg.py ^
--content_path "D:\MyProject\Dataset\coco2017\train2017" ^
--style_path "D:\MyProject\Dataset\wikiartforclassify" ^
--name artvgg19_2200_24class_with_norm ^
--model artvgg ^
--dataset_mode classify ^
--no_dropout ^
--load_size 256 ^
--crop_size 224 ^
--gpu_ids 0 ^
--batch_size 56 ^
--n_epochs 20 ^
--n_epochs_decay 3 ^
--display_freq 10 ^
--display_port 8097 ^
--display_env ArtVGG ^
--save_latest_freq 1000 ^
--data_norm

