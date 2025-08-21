#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

// ReLU activation
double relu(double x)
{
    return x > 0 ? x : 0;
}

int main()
{
    int input_size;
    cin >> input_size;
    int no_hidden_layer;
    cin >> no_hidden_layer;

    // Input to first hidden layer weights
    vector<vector<double>> weight(no_hidden_layer, vector<double>(input_size, 0));
    for (int i = 0; i < no_hidden_layer; i++)
    {
        for (int j = 0; j < input_size; j++)
        {
            double u;
            cin >> u;
            weight[i][j] = u;
        }
    }

    // Bias for first hidden layer
    vector<double> bias1(no_hidden_layer, 0);
    for (int i = 0; i < no_hidden_layer; i++)
    {
        double b;
        cin >> b;
        bias1[i] = b;
    }

    // Input vector
    vector<double> input(input_size, 0);
    for (int i = 0; i < input_size; i++)
    {
        double u;
        cin >> u;
        input[i] = u;
    }

    // First hidden layer output
    vector<double> output(no_hidden_layer, 0);
    for (int i = 0; i < no_hidden_layer; i++)
    {
        double sum = 0;
        for (int j = 0; j < input_size; j++)
        {
            sum += weight[i][j] * input[j];
        }
        output[i] = relu(sum + bias1[i]);
    }

    for (int i = 0; i < no_hidden_layer; i++)
    {
        cout << output[i] << " ";
    }

    // Number of hidden layers in total
    int total_hidden_layers;
    cin >> total_hidden_layers;

    // Hidden layer to hidden layer weights
    vector<vector<vector<double>>> weight_input(total_hidden_layers - 1, vector<vector<double>>(no_hidden_layer, vector<double>(no_hidden_layer, 0)));
    for (int i = 0; i < total_hidden_layers - 1; i++)
    {
        for (int j = 0; j < no_hidden_layer; j++)
        {
            for (int k = 0; k < no_hidden_layer; k++)
            {
                double u;
                cin >> u;
                weight_input[i][j][k] = u;
            }
        }
    }

    // Biases for each hidden layer
    vector<vector<double>> bias_hidden(total_hidden_layers - 1, vector<double>(no_hidden_layer, 0));
    for (int i = 0; i < total_hidden_layers - 1; i++)
    {
        for (int j = 0; j < no_hidden_layer; j++)
        {
            double b;
            cin >> b;
            bias_hidden[i][j] = b;
        }
    }

    // Hidden layer outputs across all layers
    vector<vector<double>> hidden_output(total_hidden_layers, vector<double>(no_hidden_layer, 0));

    // first layer already computed
    for (int i = 0; i < no_hidden_layer; i++)
    {
        hidden_output[0][i] = output[i];
    }

    // propagate through subsequent hidden layers
    for (int l = 1; l < total_hidden_layers; l++)
    {
        for (int j = 0; j < no_hidden_layer; j++)
        {
            double sum = 0;
            for (int k = 0; k < no_hidden_layer; k++)
            {
                sum += weight_input[l - 1][j][k] * hidden_output[l - 1][k];
            }
            hidden_output[l][j] = relu(sum + bias_hidden[l - 1][j]);
        }
    }

    cout << endl;
    // print final hidden layer output
    for (int i = 0; i < no_hidden_layer; i++)
    {
        cout << hidden_output[total_hidden_layers - 1][i] << " ";
    }

    //  Final Output Layer
    int total_output;
    cin >> total_output;

    // Weights from last hidden layer -> output
    vector<vector<double>> final_weight(total_output, vector<double>(no_hidden_layer, 0));
    for (int i = 0; i < total_output; i++)
    {
        for (int j = 0; j < no_hidden_layer; j++)
        {
            double u;
            cin >> u;
            final_weight[i][j] = u;
        }
    }

    // Bias for output layer
    vector<double> bias_out(total_output, 0);
    for (int i = 0; i < total_output; i++)
    {
        double b;
        cin >> b;
        bias_out[i] = b;
    }

    vector<double> final_input = hidden_output.back();
    vector<double> final_output(total_output, 0);

    for (int i = 0; i < total_output; i++)
    {
        double sum = 0;
        for (int j = 0; j < no_hidden_layer; j++)
        {
            sum += final_weight[i][j] * final_input[j];
        }
        final_output[i] = sum + bias_out[i];
    }

    cout << endl;
    // Print final output layer result
    for (int i = 0; i < total_output; i++)
    {
        cout << final_output[i] << " ";
    }

    cout << endl;
}