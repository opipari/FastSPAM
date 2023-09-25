EXPERIMENT_NAME="train_r50"
OUTPUT_DIR="results"

echo "Testing nvcc before toolkit install"
echo $(nvcc -C)

echo "Installing nvidia toolkit"
sudo apt install nvidia-cuda-toolkit 

echo "Testing nvcc after toolkit install"
echo $(nvcc -C)

echo "Setting up virtualenvironment"
# Setup virtualenvironment
python3.8 -m venv ./envs/paot-benchmark
source ./envs/paot-benchmark/bin/activate
pip install -r ./requirements/paot-benchmark/base.txt
pip install -r ./requirements/paot-benchmark/deps.txt



echo "Finished"