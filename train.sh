for s in 9
do
	# Training Proposed Method
#	python main.py --model coatnet_0 --lr 0.001 --seed ${s} --gpu 0 --dataset CIFAR100 --data_path ../dataset
	# Training Baseline
	python main.py --model t2t --is_SCL --lr 0.001 --seed ${s} --gpu 0 --dataset CIFAR100 --data_path ../dataset
done
