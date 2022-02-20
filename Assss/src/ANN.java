import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class ANN {

    //function for reading the file into double ARRay list
    public static ArrayList<Double> readFileInList(String fileName) throws FileNotFoundException {

        File file = new File(fileName);
        Scanner sc = new Scanner(file);
        String line;
        ArrayList<Double> list = new ArrayList();
        while (sc.hasNext()){
            line = sc.next();
            Double userInputAsDouble = Double.valueOf(line);
            list.add(userInputAsDouble);

        }

        return list;
    }
    public static void WriteWeightsToFile(double[] hWeight, double[] oWeight) {

        try {
            File myObj = new File("Weights.txt");
            if (myObj.createNewFile()) {
//                System.out.println("File created: " + myObj.getName());
            } else {
//                System.out.println("File already exists.");
            }
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
        try {
            FileWriter myWriter = new FileWriter("Weights.txt");
            for (int i = 0; i < hWeight.length ; i++) {
                myWriter.write(String.valueOf(hWeight[i]) + " " );
            }
            for (int i = 0; i < oWeight.length ; i++) {
                myWriter.write(String.valueOf(oWeight[i]) + " " );
            }

            myWriter.close();
            System.out.println("Successfully weights have been wrote to the file.");
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }

    public static double MSE(double[] output , double[] actualOutput, double Mse){
        for (int i = 0; i < output.length ; i++) {
            Mse += Math.pow(output[i] - actualOutput[i],2);
        }
        Mse /=  output.length;

        return Mse;
    }

    public static ArrayList<Double> mean(ArrayList<ArrayList<Double>> Arr) {
        double sum = 0;
        ArrayList<Double> meanForAllColumns = new ArrayList<>();
        for (int i = 0; i < Arr.size(); i++) {
            for (int j = 0; j < Arr.get(i).size() ; j++) {
                sum += Arr.get(i).get(j);
            }
            meanForAllColumns.add(sum/Arr.get(i).size());
        }
        return meanForAllColumns;
    }

    public static ArrayList<Double> StdDev(ArrayList<ArrayList<Double>> Arr , ArrayList<Double> mean) {
        double sum = 0;
        double standardDeviation = 0;
        double temp = 0;
        double finalResult = 0;
        ArrayList<Double> STDForAllColumns = new ArrayList<>();

        for (int i = 0; i < Arr.size(); i++) {
            for (int j = 0; j < Arr.get(i).size() ; j++) {
                standardDeviation = standardDeviation + Math.pow((Arr.get(i).get(j) - mean.get(i)), 2);
            }
            temp = standardDeviation / Arr.get(i).size();
            finalResult = Math.sqrt(temp);
            STDForAllColumns.add(finalResult);
        }

        return STDForAllColumns;
    }

    public static void ActivationFunction(double arr[]){
        for (int i = 0; i < arr.length ; i++) {
            arr[i] = 1/(1 + Math.exp(-arr[i]));
        }
    }

    public static double FF( ArrayList<ArrayList<Double>> features,ArrayList<ArrayList<Double>> outputs, int row, double[] input, double[] hidden,
                           double[] output, double[] actualOutput, double[] hWeight, double[] oWeight, double Mse ){

        for (int i = 0; i < input.length - 1 ; i++) {
            input[i] = features.get(i).get(row);
        }
        for (int i = 0; i < actualOutput.length ; i++) {
            actualOutput[i] = outputs.get(i).get(row);
        }

        // values of a(h)
        for (int i = 0; i < hidden.length  ; i++) {
            for (int j = 0; j < input.length ; j++) {
                hidden[i] += input[j] * hWeight[ j * hidden.length + i];
            }
        }
        ActivationFunction(hidden);

        //values of a(o)
        for (int i = 0; i < output.length  ; i++) {
            for (int j = 0; j < hidden.length ; j++) {
                output[i] += hidden[j] * oWeight[ j * output.length + i];
            }
        }
        ActivationFunction(output);
        return MSE(output,actualOutput,Mse);

    }
    public static void BP(double[] input, double[] hidden, double[] output,
                          double[] actualOutput, double[] oWeight, double[]hWeight, int row ){

        double learningRate = 0.03;



        //calculating sigma(O)
        double[] sigmaO = new double[output.length];
        double[] sigmaH = new double[hidden.length];
        for (int i = 0; i < output.length ; i++) {
            sigmaO[i] = (output[i] - actualOutput[i]) * output[i] * (1-output[i]);
        }
        //calculating sigma(h)
        for (int i = 0; i < hidden.length ; i++) {
            for (int j = 0; j < output.length ; j++) {
                sigmaH[i] = sigmaO[j] * oWeight[j * output.length + i];
            }
            sigmaH[i] *= hidden[i] * (1 - hidden[i] );
        }
        //calculating and updating new oWeights
        for (int i = 0; i < hidden.length ; i++) {
            for (int j = 0; j < output.length ; j++) {
                oWeight[output.length * i + j] = oWeight[output.length * i + j] - learningRate * sigmaO[j] * hidden[i];
            }
        }
        //calculating and updating new hWeights
        for (int i = 0; i < input.length ; i++) {
            for (int j = 0; j < hidden.length ; j++) {
                hWeight[hidden.length * i + j] = hWeight[hidden.length * i + j] - learningRate * sigmaH[j] * input[i];
            }
        }
    }



    public static void main(String args[]) throws FileNotFoundException {

        ArrayList<Double> dataList =  readFileInList("./ass4 data.txt");

        //splitting the data into features , every feature in a column like python
        ArrayList<ArrayList<Double>> features = new ArrayList<>();
        for (int i = 0; i < dataList.get(0) ; i++) {
            ArrayList<Double> inputTemp = new ArrayList<>();
            features.add(inputTemp);
        }

        ArrayList<ArrayList<Double>> outputs = new ArrayList<>();
        for (int i = 0; i < dataList.get(2) ; i++) {
            ArrayList<Double> outputTemp = new ArrayList<>();
            outputs.add(outputTemp);
        }


        for (int i = 4; i < dataList.get(3) * (dataList.get(0) + dataList.get(2) ) ; i+= (dataList.get(0) + dataList.get(2) ) ) {

           int inputSizeTemp = 0;
           int outputSizeTemp = 0;
           while (inputSizeTemp != dataList.get(0)){
                features.get(inputSizeTemp).add(dataList.get(i + inputSizeTemp));
                inputSizeTemp++;
            }
           while (outputSizeTemp != dataList.get(2)){
               outputs.get(outputSizeTemp).add(dataList.get(i + outputSizeTemp));
               outputSizeTemp++;
           }

        }
        //end of splitting data

        //normalizing all the features and all the outputs into new arraylists

        //Mean
        ArrayList<Double> FeaturesMeans = mean(features);
        ArrayList<Double> OutputsMeans = mean(outputs);
        //std dev
        ArrayList<Double> FeaturesSTDs = StdDev(features , FeaturesMeans);
        ArrayList<Double> OutputsSTDs = StdDev(outputs , OutputsMeans);
        //Normalized Data (features , outputs)
        for (int i = 0; i < features.size() ; i++) {
            for (int j = 0; j < features.get(i).size() ; j++) {
                features.get(i).set(j , ( features.get(i).get(j) - FeaturesMeans.get(i) ) / FeaturesSTDs.get(i) );
            }
        }

        for (int i = 0; i < outputs.size() ; i++) {
            for (int j = 0; j < outputs.get(i).size() ; j++) {
                outputs.get(i).set(j , ( outputs.get(i).get(j) - OutputsMeans.get(i) ) / OutputsSTDs.get(i) );
            }
        }

        System.out.println("preprocessing data has been Accomplished");
        // establishing the Neural Network
        Double inpSize = dataList.get(0) + 1;
        Double hidSize = dataList.get(1);
        Double outSize = dataList.get(2);
        int sizeOfInputs = inpSize.intValue();
        int sizeOfHiddens = hidSize.intValue();
        int sizeOfOutputs = outSize.intValue();

        double inputNodes[] = new double[sizeOfInputs];
        double hiddenNodes[] = new double[sizeOfHiddens];
        double predictedOutputNodes[] = new double[sizeOfOutputs];
        double actualOutputNodes[] = new double[sizeOfOutputs];

        double weightsHidden[] = new double[sizeOfInputs * sizeOfHiddens];
        double weightsOutput[] = new double[sizeOfHiddens * sizeOfOutputs];
        double MSE = 0;

        Random r = new Random();
        for (int i = 0; i < weightsHidden.length ; i++) {
            double randomValue = -2 + (2 + 2) * r.nextDouble();
            weightsHidden[i] = randomValue;
        }
        for (int i = 0; i < weightsOutput.length ; i++) {
            double randomValue = -2 + (2 + 2) * r.nextDouble();
            weightsOutput[i] = randomValue;
        }
        //Fitting the Model
        // iterations for training
        int iteration = 1000;
        for (int i = 0; i < iteration ; i++) {
            int row = 0;
            // examples to be used for training
            for (int j = 0; j < dataList.get(3) ; j++) {
                //bias in inputs
                inputNodes[inputNodes.length-1] = 1;
                //feed forward
                MSE = FF( features,outputs, row, inputNodes, hiddenNodes, predictedOutputNodes,actualOutputNodes,weightsHidden, weightsOutput, MSE );
//                System.out.println("FF has been Accomplished");
                //Back Propagation
                BP(inputNodes, hiddenNodes, predictedOutputNodes, actualOutputNodes, weightsOutput, weightsHidden, row );
//                System.out.println("BP has been Accomplished");
                Arrays.fill(hiddenNodes,0);
                Arrays.fill(predictedOutputNodes,0);
                row++;

            }
            System.out.println(MSE);
            if (i !=iteration-1)
                MSE = 0;

        }
        System.out.println("Training has been Accomplished");
        System.out.println("MSE for each example after specified number of iteration =" +
                " MSE for all examples in the last iteration / number of examples" + "= " + MSE/dataList.get(3));

        WriteWeightsToFile(weightsHidden,weightsOutput);


    }

}
