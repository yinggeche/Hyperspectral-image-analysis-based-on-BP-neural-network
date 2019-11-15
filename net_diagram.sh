
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

img_type=png
img_fileprefix=ResNet_50
conf_filename=trainer_config.py
dot_filename=Net.dot
config_str="layer_num=4,data_provider=0"

python -m paddle.utils.make_model_diagram $conf_filename $dot_filename $config_str

dot -Tpng -o Net.png Net.dot
