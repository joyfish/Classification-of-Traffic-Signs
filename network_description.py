from parameters import *

print('\nDeep Convolutional Neural Network is structured as followed:')
print('============================================================\n')
print('-> Convolutional Layer 1\tInput size: {}x{}\tInput Depth: {}\t\tFilter size: {}x{}\tOutput size: {}x{}\tOutput Depth: {}'.format(height, width, n_colors, filters['hw_1'], filters['hw_1'], output_dims['c_1'], output_dims['c_1'], depths_out['c_1']))
print('-> Pooling Layer 1\t\tInput size: {}x{}\tInput Depth: {}\t\tFilter size: {}x{}\tOutput size: {}x{}\tOutput Depth: {}'.format(output_dims['c_1'], output_dims['c_1'], depths_out['c_1'], pool_dim, pool_dim, output_dims['p_1'], output_dims['p_1'], depths_out['c_1']))
print('-> ReLU Activation Function')
print('-> Convolutional Layer 2\tInput size: {}x{}\tInput Depth: {}\t\tFilter size: {}x{}\tOutput size: {}x{}\tOutput Depth: {}'.format(output_dims['p_1'], output_dims['p_1'], depths_out['c_1'], filters['hw_2'], filters['hw_2'], output_dims['c_2'], output_dims['c_2'], depths_out['c_2']))
print('-> Pooling Layer 2\t\tInput size: {}x{}\tInput Depth: {}\tFilter size: {}x{}\tOutput size: {}x{}\tOutput Depth: {}'.format(output_dims['c_2'], output_dims['c_2'], depths_out['c_2'], pool_dim, pool_dim, output_dims['p_2'], output_dims['p_2'], depths_out['c_2']))
print('-> ReLU Activation Function')
print('-> Convolutional Layer 3\tInput size: {}x{}\t\tInput Depth: {}\tFilter size: {}x{}\tOutput size: {}x{}\tOutput Depth: {}'.format(output_dims['p_2'], output_dims['p_2'], depths_out['c_2'], filters['hw_3'], filters['hw_3'], output_dims['c_3'], output_dims['c_3'], depths_out['c_3']))
print('-> Pooling Layer 3\t\tInput size: {}x{}\t\tInput Depth: {}\tFilter size: {}x{}\tOutput size: {}x{}\tOutput Depth: {}'.format(output_dims['c_3'], output_dims['c_3'], depths_out['c_3'], pool_dim, pool_dim, output_dims['p_3'], output_dims['p_3'], depths_out['c_3']))
print('-> ReLU Activation Function')
print('-> Fully Connected Layer 1\tNeurons: {}\t\tInputs: {}\t\tOutputs: {}'.format(layer_size['fc_1'], output_dims['p_3']*output_dims['p_3']*depths_out['c_3'], layer_size['fc_1']))
print('-> Fully Connected Layer 2\tNeurons: {}\t\tInputs: {}\t\tOutputs: {}'.format(layer_size['fc_2'], layer_size['fc_1'],layer_size['fc_2']))
print('-> Output Layer\t\t\tNeurons: {}\t\tInputs: {}\t\tOutputs: {}'.format(n_classes, layer_size['fc_2'], n_classes))
print('-> Softmax Function\n')