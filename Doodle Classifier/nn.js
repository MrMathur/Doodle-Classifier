var NeuralNetwork = function(input_num, hidden_num, output_num, lr) {
    this.input_num = input_num;
    this.hidden_num = hidden_num;
    this.output_num = output_num;

    this.lr = lr;

    this.bias_hi = [];
    for (var i=0; i<hidden_num; i++) {
        this.bias_hi[i] = [random(-1,1)];
    }
    this.bias_oh = [];
    for (var i=0; i<output_num; i++) {
        this.bias_oh[i] = [random(-1,1)];
    }

    this.weights_hi = [];
    for (var i=0; i<this.hidden_num; i++) {
        this.weights_hi[i] = [];
        for (var j=0; j<this.input_num; j++) {
            this.weights_hi[i][j] = random(-1,1);
        }
    }
    this.weights_oh = [];
    for (var i=0; i<this.output_num; i++) {
        this.weights_oh[i] = [];
        for (var j=0; j<this.hidden_num; j++) {
            this.weights_oh[i][j] = random(-1,1);
        }
    }

    this.predict = function(inputs) {
        if (inputs.length === this.input_num) {
            var hidden_matrix = Matrix_Multiplication(this.weights_hi, inputs);
            hidden_matrix = Matrix_Addition(hidden_matrix, this.bias_hi);
            hidden_matrix = Matrix_Function(hidden_matrix, Sigmoid);

            var output_matrix = Matrix_Multiplication(this.weights_oh, hidden_matrix);
            output_matrix = Matrix_Addition(output_matrix, this.bias_oh);
            output_matrix = Matrix_Function(output_matrix, Sigmoid);

            return output_matrix;
        } else {
            return -1;
        }
    }

    this.train = function(inputs, targets) {
        if (inputs.length === this.input_num) {
            var hidden_matrix = Matrix_Multiplication(this.weights_hi, inputs);
            hidden_matrix = Matrix_Addition(hidden_matrix, this.bias_hi);
            hidden_matrix = Matrix_Function(hidden_matrix, Sigmoid);

            var output_matrix = Matrix_Multiplication(this.weights_oh, hidden_matrix);
            output_matrix = Matrix_Addition(output_matrix, this.bias_oh);
            output_matrix = Matrix_Function(output_matrix, Sigmoid);

            var error_output = Matrix_Subtraction(targets, output_matrix);
            var error_hidden = Matrix_Multiplication(Matrix_Transpose(this.weights_oh), error_output);

            var delta_oh = Matrix_Function(output_matrix, SigmoidDeriv);
            delta_oh = Matrix_Elem_Multiplication(error_output, delta_oh);
            delta_oh = Matrix_Scalar_Multiplication(delta_oh, this.lr);

            var delta_bias_oh = delta_oh;
            delta_oh = Matrix_Multiplication(delta_oh, Matrix_Transpose(hidden_matrix));
            
            var delta_hi = Matrix_Function(hidden_matrix, SigmoidDeriv);
            delta_hi = Matrix_Elem_Multiplication(error_hidden, delta_hi);
            delta_hi = Matrix_Scalar_Multiplication(delta_hi, this.lr);
            
            var delta_bias_hi = delta_hi;
            delta_hi = Matrix_Multiplication(delta_hi, Matrix_Transpose(inputs));

            this.weights_hi = Matrix_Addition(this.weights_hi, delta_hi);
            this.weights_oh = Matrix_Addition(this.weights_oh, delta_oh);

            this.bias_oh = Matrix_Addition(this.bias_oh, delta_bias_oh);
            this.bias_hi = Matrix_Addition(this.bias_hi, delta_bias_hi);
        } else {
            return -1;
        }
    }
}