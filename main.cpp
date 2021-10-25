#include <fstream>
#include <iostream>
#include <math.h>
#include <time.h>

class Data {
public:
    long double* input;
    long double* target;
    int input_size;
    int target_size;
    Data(long double* input, int input_size, long double* target, int target_size)
    {
        this->input = input;
        this->input_size = input_size;
        this->target = target;
        this->target_size = target_size;
    }
    ~Data()
    {
        delete input;
        delete target;
    }
};

class ImageData {
public:
    int** pixel;
    int y;
    int cols;
    int rows;
    ImageData(int** pixel, int y, int cols, int rows)
    {
        this->pixel = pixel;
        this->y = y;
        this->cols = cols;
        this->rows = rows;
    }
    ~ImageData()
    {
        //std::cout << "destructor called" << std::endl;
        for (int i = 0; i < rows; i++) {
            delete pixel[i];
        }
        delete pixel;
    }
};

class Network {
public:
    // also the number of layers
    int num_neuron_size;
    /*
            contains the number of neurons for every layer
        */
    int* num_neurons;
    /*
            contains weight from b to c. weight[a][b][c]
            layer starts at 0
            a = neurons between layer a to a+1. size of num_neuron_size-1
            b = neuron in layer a. size of num_neurons[a]
            c = neuron in layer a+1. size of num_neurons[a+1]
        */
    long double*** weight;
    /*
            bias[a][b].
            a = layer a+1. size of num_neuron_size-1
            b = neuron b in layer. size of num_neurons[a+1]
        */
    long double** bias;
    /*
        output[a][b]. used for calculating backprop of data
        layer a, node b
    */
    long double** output;
    /*
        delta_weight[a][b][c] used for calculating delta weights
            a = neurons between layer a to a+1. size of num_neuron_size-1
            b = neuron in layer a. size of num_neurons[a]
            c = neuron in layer a+1. size of num_neurons[a+1]
    */
    long double*** delta_weight;
    /*
            delta_bias[a][b].
            a = layer a+1. size of num_neuron_size-1
            b = neuron b in layer. size of num_neurons[a+1]
        */
    long double** delta_bias;
    /*
        cost_derivative[a][b]. used for saving partial derivatives up to sigmoid
        layer a+1, node b
    */
    long double** cost_derivative;
    long double learning_rate = 1;
    Network(int* num_neurons, int num_neuron_size, long double learning_rate)
    {
        this->num_neurons = num_neurons;
        this->num_neuron_size = num_neuron_size;
        this->learning_rate = learning_rate;

        weight = new long double**[num_neuron_size - 1];
        delta_weight = new long double**[num_neuron_size - 1];
        for (int i = 0; i < num_neuron_size - 1; i++) {
            weight[i] = new long double*[num_neurons[i]];
            delta_weight[i] = new long double*[num_neurons[i]];
            for (int j = 0; j < num_neurons[i]; j++) {
                weight[i][j] = new long double[num_neurons[i + 1]];
                delta_weight[i][j] = new long double[num_neurons[i + 1]];
                for (int k = 0; k < num_neurons[i + 1]; k++) {
                    weight[i][j][k] = gaussianRandom();
                }
            }
        }
        

        bias = new long double*[num_neuron_size - 1];
        delta_bias = new long double*[num_neuron_size - 1];
        cost_derivative = new long double*[num_neuron_size - 1];
        for (int i = 0; i < num_neuron_size - 1; i++) {
            bias[i] = new long double[num_neurons[i + 1]];
            delta_bias[i] = new long double[num_neurons[i + 1]];
            cost_derivative[i] = new long double[num_neurons[i + 1]];
            for (int j = 0; j < num_neurons[i + 1]; j++) {
                bias[i][j] = 0;
            }
        }

        output = new long double*[num_neuron_size];
        for (int j = 0; j < num_neuron_size; j++) {
            output[j] = new long double[num_neurons[j]];
        }
    }

    void addData(Data** mini_batch, int mini_batch_size)
    {
        //initialize values
        long double totalerror = 0;
        for (int i = 0; i < num_neuron_size - 1; i++) {
            //node j in layer i
            for (int j = 0; j < num_neurons[i]; j++) {
                //node k in layer i+1
                for (int k = 0; k < num_neurons[i + 1]; k++) {
                    delta_weight[i][j][k] = 0;
                }
            }
        }
        //layer i+1
        for (int i = 0; i < num_neuron_size - 1; i++) {
            //node j in layer i
            for (int j = 0; j < num_neurons[i+1]; j++) {
                delta_bias[i][j] = 0;
            }
        }

        for (int mini_batch_index = 0; mini_batch_index < mini_batch_size; mini_batch_index++) {
            //initialize cost_derivative
            for (int i = 0; i < num_neuron_size - 1; i++) {
                //node j in layer i
                for (int j = 0; j < num_neurons[i+1]; j++) {
                    cost_derivative[i][j] = 0;
                }
            }
            //forward
            long double* target = mini_batch[mini_batch_index]->target;
            output[0] = mini_batch[mini_batch_index]->input;
            //layer j
            for (int j = 1; j < num_neuron_size; j++) {
                //node k in the layer j
                for (int k = 0; k < num_neurons[j]; k++) {
                    output[j][k] = 0;
                    //node o in layer j-1
                    for (int o = 0; o < num_neurons[j-1]; o++) {
                        output[j][k] += output[j-1][o] * weight[j-1][o][k];
                    }
                    output[j][k] += bias[j-1][k];
                    output[j][k] = sigmoid(output[j][k]);
                }
            }

            //calculate total error
            for (int j = 0; j < num_neurons[num_neuron_size-1]; j++) {
                long double diff = output[num_neuron_size-1][j] - target[j];
                totalerror += diff*diff/2;
            }

            //backwards
            //node i in last layer num_neuron_size-1
            for (int i=0;i<num_neurons[num_neuron_size-1];i++) {
                cost_derivative[num_neuron_size-2][i] = output[num_neuron_size-1][i] * (1-output[num_neuron_size-1][i]) * (output[num_neuron_size-1][i]-target[i]);
                delta_bias[num_neuron_size-2][i] += cost_derivative[num_neuron_size-2][i];
                //node j in layer num_neuron_size-2
                for (int j=0;j<num_neurons[num_neuron_size-2];j++) {
                    delta_weight[num_neuron_size-2][j][i] += cost_derivative[num_neuron_size-2][i] * output[num_neuron_size-2][j];
                }
            }
            //layer i
            for (int i=num_neuron_size-3;i>-1;i--){
                //node j in layer i+1
                for (int j=0;j<num_neurons[i+1];j++) {
                    cost_derivative[i][j] = 0;
                    //node m in layer i+2
                    for (int m=0; m<num_neurons[i+2];m++) {
                        cost_derivative[i][j] += cost_derivative[i+1][m]*weight[i+1][j][m];
                    }
                    cost_derivative[i][j] *= output[i+1][j]*(1-output[i+1][j]);
                    //node k in layer i
                    for (int k=0;k<num_neurons[i];k++) {

                        delta_weight[i][k][j] += cost_derivative[i][j] * output[i][k];
                    }
                    delta_bias[i][j] += cost_derivative[i][j];
                }
            }
        }

        //modify weights
        //layer i
        for (int i = 0; i < num_neuron_size - 1; i++) {
            //node j in layer i
            for (int j = 0; j < num_neurons[i]; j++) {
                //node k in layer i+1
                for (int k = 0; k < num_neurons[i + 1]; k++) {
                    weight[i][j][k] -= learning_rate*(delta_weight[i][j][k]/mini_batch_size);
                }
            }
        }
        //modify biases
        //layer i+1
        for (int i = 0; i < num_neuron_size - 1; i++) {
            //node j in layer i
            for (int j = 0; j < num_neurons[i+1]; j++) {
                bias[i][j] -= learning_rate*(delta_bias[i][j]/mini_batch_size);
            }
        }
    }

    long double cost(Data* d) {
        
        //forward
        //layer j
        for (int j = 1; j < num_neuron_size; j++) {
            //node k in the layer j
            for (int k = 0; k < num_neurons[j]; k++) {
                output[j][k] = 0;
                //node o in layer j-1
                for (int o = 0; o < num_neurons[j-1]; o++) {
                    output[j][k] += output[j-1][o] * weight[j-1][o][k];
                }
                output[j][k] += bias[j-1][k];
                output[j][k] = sigmoid(output[j][k]);
            }
        }

        long double totalerror = 0;

        //calculate total error
        for (int j = 0; j < num_neurons[num_neuron_size-1]; j++) {
            long double diff = output[num_neuron_size-1][j] - d->target[j];
            totalerror += diff*diff/2;
        }

        return totalerror;
    }

    bool guess(Data* d) {
        long double totalerror = 0;
        //forward
        output[0] = d->input;
        //layer j
        for (int j = 1; j < num_neuron_size; j++) {
            //node k in the layer j
            for (int k = 0; k < num_neurons[j]; k++) {
                output[j][k] = 0;
                //node o in layer j-1
                for (int o = 0; o < num_neurons[j-1]; o++) {
                    output[j][k] += output[j-1][o] * weight[j-1][o][k];
                }
                output[j][k] += bias[j-1][k];
                output[j][k] = sigmoid(output[j][k]);
            }
        }
        long double bestscore=0;
        int bestnum=0;
        long double besttarget=0;
        int targetnum=0;
        for (int j = 0; j < num_neurons[num_neuron_size-1]; j++) {
            if (output[num_neuron_size-1][j] > bestscore) {
                bestscore = output[num_neuron_size-1][j];
                bestnum = j;
            }
            if (d->target[j] > besttarget) {
                besttarget = d->target[j];
                targetnum = j;
            }
        }
        if (bestnum == targetnum) return true;
        return false;
    }
    
    int guess(Data** d, int d_length) {
        int correct = 0;
        for (int i=0;i<d_length;i++) {
            if (guess(d[i])) correct++;
        }
        return correct;
    }

    void save() {
        std::ofstream networkfile;
        networkfile.open ("weight.txt");

        networkfile << "["; 
        for (int i = 0; i < num_neuron_size - 1; i++) {
            networkfile << "["; 
            for (int k = 0; k < num_neurons[i]; k++) {
                networkfile << "["; 
                for (int j = 0; j < num_neurons[i+1]; j++) {
                    networkfile << weight[i][k][j];
                    if (j < num_neurons[i+1]-1) networkfile << ",";
                }
                networkfile << "]";
                if (k < num_neurons[i]-1) networkfile << ",";
            }
            networkfile << "]";
            if (i < num_neuron_size-2) networkfile << ",";
        }
        networkfile << "]";
        networkfile.close();

        std::ofstream networkfile2;
        networkfile2.open ("bias.txt");

        networkfile2 << "["; 
        for (int i = 0; i < num_neuron_size - 1; i++) {
            networkfile2 << "["; 
            for (int j = 0; j < num_neurons[i+1]; j++) {
                networkfile2 << bias[i][j];
                if (j < num_neurons[i+1]-1) networkfile2 << ",";
            }
            networkfile2 << "]";
            if (i < num_neuron_size-2) networkfile2 << ",";
        }
        networkfile2 << "]";
        networkfile2.close();
    }

    ~Network()
    {
        //delete weights and biases
    }

private:
    //not really but good enough
    long double gaussianRandom()
    {
        long double uni_random = 0;
        for (int a = 0; a < 12; a++) {
            uni_random += rand();
        }
        return (long double)uni_random / RAND_MAX - 6;
    }
    long double sigmoid(long double x)
    {
        return 1 / (1 + exp(-x));
    }
};

//https://www.nist.gov/itl/products-and-services/emnist-dataset
class MNIST {
public:
    std::ifstream imagesfile;
    std::ifstream labelsfile;
    int num_images=0;
    int num_rows=0;
    int num_cols=0;
    int cur = 0;
    MNIST(char* imagefilename, char* labelfilename)
    {
        imagesfile.open(imagefilename, std::ios::in | std::ios::binary);
        labelsfile.open(labelfilename, std::ios::in | std::ios::binary);
        char c;
        char c2;
        unsigned char imagesheader[4] = { 0x00, 0x00, 0x08, 0x03 };
        unsigned char labelsheader[4] = { 0x00, 0x00, 0x08, 0x01 };
        for (int i = 0; i < 4; i++) {
            imagesfile.get(c);
            if ((unsigned char)c != imagesheader[i])
                throw 20;
            labelsfile.get(c);
            if ((unsigned char)c != labelsheader[i])
                throw 30;
        }
        for (int i = 0; i < 4; i++) {
            imagesfile.get(c);
            labelsfile.get(c2);
            if (c != c2)
                throw 40;
            num_images = (num_images << 8) + (int)(unsigned char)c;
        }
        for (int i = 0; i < 4; i++) {
            imagesfile.get(c);
            num_rows = (num_rows << 8) + (int)(unsigned char)c;
        }
        for (int i = 0; i < 4; i++) {
            imagesfile.get(c);
            num_cols = (num_cols << 8) + (int)(unsigned char)c;
        }
    }

    ImageData* getimage()
    {
        if (cur < this->num_images) {
            char c;
            labelsfile.get(c);
            int output = (int)(unsigned char)c;
            int** pixel = new int*[num_rows];
            for (int i = 0; i < num_rows; i++) {
                pixel[i] = new int[num_cols];
                for (int j = 0; j < num_cols; j++) {
                    imagesfile.get(c);
                    pixel[i][j] = (int)(unsigned char)c;
                }
            }
            cur++;
            return new ImageData(pixel, output, num_rows, num_cols);
        } else {
            std::cout << "cur: " << cur << std::endl;
            std::cout << "num_images: " << num_images << std::endl;
            throw 50;
        }
    }

    Data* getData() {
        ImageData* img = getimage();
        //std::cout << "2cur: " << m->cur << std::endl;
        long double* input = new long double[img->rows*img->cols];
        long double* target = new long double[10];
        for (int k=0;k<10;k++) {
            target[k] = 0;
        }
        target[img->y] = 1;
        for (int k = 0; k < img->rows; k++) {
            for (int m = 0; m < img->cols; m++) {
                //long double x = (long double)img->pixel[k][m]/255;
                //the data is flipped for some reason      
                long double x = (long double)img->pixel[m][k]/255;
                input[k * img->cols + m] = x;
            }
        }
        delete img;
        return new Data(input, img->rows*img->cols, target, 10);
    }

    ~MNIST() {
        imagesfile.close();
        labelsfile.close();
    }
};
int main()
{
    //srand(time(0));
    srand(0);
    try {
        std::cout.precision(100);
        std::fixed;
        //MNIST* m = new MNIST("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
        MNIST* m = new MNIST("emnist-digits-train-images-idx3-ubyte", "emnist-digits-train-labels-idx1-ubyte");
        std::cout << m->num_images << std::endl;
        std::cout << m->num_rows << std::endl;
        std::cout << m->num_cols << std::endl;

        const int layers = 4;
        //int neurons[layers] = { 784, 500, 10 };
        int neurons[layers] = { 784, 50, 50, 10 };
        Network* n = new Network(neurons, layers, 1);
        int totalimages = m->num_images;
        
        const int testdata_length = 1000;
        totalimages -= testdata_length;
        Data** testdata = new Data*[testdata_length];
        for (int i = 0; i < testdata_length; i++) {
            testdata[i] = m->getData();
        }
        const int mini_batch_size = 10;
        Data* batch[mini_batch_size];
        //mini batch i
        int num_batches = 100;
        int num_epoches = totalimages/mini_batch_size/num_batches;
        std::cout << "correct: " << n->guess(testdata,testdata_length) << "/" << testdata_length << std::endl;
        for (int k = 0; k < num_epoches; k++) {
            for (int i = 0; i < num_batches; i++) {
                //image j in minibatch
                for (int j = 0; j < mini_batch_size; j++) {
                    Data* data = m->getData();
                    //print number
                    /*
                    for (int j=0;j<10;j++) {
                        if (data->target[j] == 1) std::cout << j << std::endl;                            
                    }
                    for (int i=0;i<784;i++) {
                        if (data->input[i]>0) {
                            int grad = (int) (data->input[i]*9);
                            std::cout << grad << grad ;
                        } else std::cout << "..";
                        if (i%28==27) std::cout << std::endl;
                    }
                    */
                    batch[j] = data;
                }
                n->addData(batch, mini_batch_size);
                for (int j = 0; j < mini_batch_size; j++) {
                    delete batch[j];
                }
            }
            std::cout << "correct: " << n->guess(testdata,testdata_length) << "/" << testdata_length << std::endl;
        }

        n->save();

    } catch (int e) {
        std::cout << "Error:" << e << std::endl;
    }
    return 0;
}