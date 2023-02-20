python train.py ^
--content_path "D:\MyProject\Dataset\coco2017\train2017" ^
--style_path "D:\MyProject\Dataset\vangogh2photo\trainA" ^
--name artvggGanSelfAdaAttN_without_artvgg_with_artmap_all_adain_with_gan_vangogh_256 ^
--model artvggganselfadaattn ^
--dataset_mode unaligned ^
--no_dropout ^
--load_size 256 ^
--crop_size 256 ^
--style_encoder_path ./checkpoints\artvgg19_2200_24class_with_norm/latest_net_artvgg.pth ^
--gpu_ids 0 ^
--batch_size 4 ^
--n_epochs 1 ^
--n_epochs_decay 0 ^
--display_freq 100 ^
--display_port 8097 ^
--display_env ArtvggSelfAdaAttN ^
--save_latest_freq 5000 ^
--lambda_local 3 ^
--lambda_global 10 ^
--lambda_content 0 ^
--lambda_edge 0 ^
--lambda_D 25 ^
--lambda_G 25 ^
--shallow_layer ^
--skip_connection_3 ^
--data_norm ^
--art_map ^
--continue_train

