# Caffe with MDLSTM extension

This fork of Caffe adds support for the Multi-Dimensional LSTM layer proposed by Alex Graves ([arXiv](https://arxiv.org/abs/0705.2011)).

The main contribution is in the `MDLSTMLayer.cpp` which implements a CPU parallel version of the layer based on OpenMP.
The code should compile without problems on Linux and MacOS (be careful though because Mac doesn't have OpenMP support by default).

Currently there is no GPU implementation for this layer and any suggestion/help is appreciated (write to 91snake91(at)gmail(dot)com).

## Example usage
The layer is called `MDLSTM` in the protobuf definition. A full example can be found in the `examples/mnist/mdlstm` folder.
Refer to the `caffe.proto` file for the full list of available params.

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
