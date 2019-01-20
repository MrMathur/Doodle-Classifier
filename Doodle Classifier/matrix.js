function Matrix_Multiplication(a,b) {
    var output = [];

    var rows = a.length;
    var cols = b[0].length;
    var seq;
    if (a[0].length === b.length) {
        seq = a[0].length;
    } else {
        console.log("Matrices can't be multiplied.");
        return -1;
    }

    for (var i=0; i<rows; i++) {
        output[i] = [];
        for (var j=0; j<cols; j++) {
            output[i][j] = 0;
            for (var k=0; k<seq; k++) {
                output[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return output;    
}

function Matrix_Elem_Multiplication(a,b) {
    var output = [];

    var rows = a.length;
    var cols = b[0].length;

    for (var i=0; i<rows; i++) {
        output[i] = [];
        for (var j=0; j<cols; j++) {
            output[i][j] = a[i][j] * b[i][j];
        }
    }

    return output;    
}

function Matrix_Scalar_Multiplication(a, k) {
    var output = [];

    var rows = a.length;
    var cols = a[0].length;

    for (var i=0; i<rows; i++) {
        output[i] = [];
        
        for (var j=0; j<cols; j++) {
            output[i][j] = a[i][j] * k;
        }
    }

    return output;   
}

function Matrix_Function(a, fn) {
    var output = [];
    
    for (var i=0; i<a.length; i++) {
        output[i] = [];
        for (var j=0; j<a[i].length; j++) {
            output[i][j] = fn(a[i][j]);
        }
    }

    return output;
}

function Matrix_Addition(a,b) {
    var output = [];

    var rows = a.length;
    var cols = b[0].length;

    for (var i=0; i<rows; i++) {
        output[i] = [];
        for (var j=0; j<cols; j++) {
            output[i][j] = a[i][j] + b[i][j];
        }
    }

    return output;    
}

function Matrix_Subtraction(a,b) {
    var output = [];

    var rows = a.length;
    var cols = b[0].length;

    for (var i=0; i<rows; i++) {
        output[i] = [];
        for (var j=0; j<cols; j++) {
            output[i][j] = a[i][j] - b[i][j];
        }
    }

    return output;    
}

function Matrix_Transpose(a) {
    var output = [];

    for (var i=0; i<a[0].length; i++) {
        output[i] = [];

        for (var j=0; j<a.length; j++) {
            output[i][j] = a[j][i];    
        }
    }

    return output;
}

function Sigmoid(x) {
    return (1/(1 + Math.exp(-1 * x)));
}

function SigmoidDeriv(x) {
    return (x * (1 - x));
}