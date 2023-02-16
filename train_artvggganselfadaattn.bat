python train.py ^
--content_path "D:\MyProject\Dataset\coco2017\train2017" ^
--style_path "D:\MyProject\Dataset\wikiart" ^
--name artvggGanSelfAdaAttN_without_artvgg_all_adain_with_gan_tt3_512 ^
--model artvggganselfadaattn ^
--dataset_mode unaligned ^
--no_dropout ^
--load_size 512 ^
--crop_size 512 ^
--style_encoder_path ./checkpoints\artvgg19_2200_24class_with_norm/latest_net_artvgg.pth ^
--gpu_ids 0 ^
--batch_size 1 ^
--n_epochs 2 ^
--n_epochs_decay 3 ^
--display_freq 200 ^
--display_port 8097 ^
--display_env ArtvggSelfAdaAttN ^
--lambda_local 3 ^
--lambda_global 10 ^
--lambda_content 0 ^
--lambda_edge 0 ^
--lambda_D 25 ^
--lambda_G 25 ^
--shallow_layer ^
--skip_connection_3 ^
--data_norm ^
--continue_train

