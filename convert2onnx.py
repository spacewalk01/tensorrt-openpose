import trt_pose.models
import torch
import json
import trt_pose.coco
import sys, getopt

def convertModel(inputFile, outputFile):

	try:
		with open('human_pose.json', 'r') as f:
		    human_pose = json.load(f)

		topology = trt_pose.coco.coco_category_to_topology(human_pose)

		num_parts = len(human_pose['keypoints'])
		num_links = len(human_pose['skeleton'])

		model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()

		MODEL_WEIGHTS = inputFile  #'resnet18_baseline_att_224x224_A_epoch_249.pth' 

		model.load_state_dict(torch.load(MODEL_WEIGHTS))

		WIDTH = 224
		HEIGHT = 224

		data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

		def default_input_names(num_inputs):
		    return ["input_%d" % i for i in range(num_inputs)]

		def default_output_names(num_outputs):
		    return ["output_%d" % i for i in range(num_outputs)]

		inputs = [data]
		inputs_in = inputs

		# copy inputs to avoid modifications to source data
		inputs = [tensor.clone()[0:1] for tensor in inputs]  # only run single entry

		if isinstance(inputs, list):
		    inputs = tuple(inputs)
		if not isinstance(inputs, tuple):
		    inputs = (inputs,)
		    
		# run once to get num outputs
		outputs = model(*inputs)
		if not isinstance(outputs, tuple) and not isinstance(outputs, list):
		    outputs = (outputs,)
		    
		input_names = default_input_names(len(inputs))
		output_names = default_output_names(len(outputs))

		torch.onnx.export(model, inputs, outputFile, input_names=default_input_names(len(inputs)), output_names=default_output_names(len(outputs)))

		print('success!')

	except:
  		print("An exception occurred")

def main(argv):

   inputfile = ''
   outputfile = ''


   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print('test.py -i <inputfile> -o <outputfile>')
      sys.exit(2)


   for opt, arg in opts:
      if opt == '-h':
         print('test.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg

   convertModel(inputfile, outputfile)


if __name__ == "__main__":
   main(sys.argv[1:])