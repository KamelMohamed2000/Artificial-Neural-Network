import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;

public class Test {




    public static void main(String[] args) throws FileNotFoundException {
        //getting test data and weights from the files
        ArrayList<Double> testList =  ANN.readFileInList("test.txt");
        ArrayList<Double> weights = ANN.readFileInList("Weights.txt");
        //spliting weights into Hidden weights and Output weights
        Double inpSize = testList.get(0) + 1;
        Double hidSize = testList.get(1);
        Double outSize = testList.get(2);
        int sizeOfInputs = inpSize.intValue();
        int sizeOfHiddens = hidSize.intValue();
        int sizeOfOutputs = outSize.intValue();

        double[] hWeights = new double[sizeOfInputs * sizeOfHiddens];
        double[] oWeights = new double[sizeOfHiddens * sizeOfOutputs];
        for (int i = 0; i < hWeights.length ; i++) {
            hWeights[i] = weights.get(i);
        }
        for (int i = 0; i < oWeights.length ; i++) {
            oWeights[i] = weights.get(i + hWeights.length);
        }
        //end of splitting weights

        //splitting the data into features , every feature in a column like python
        ArrayList<ArrayList<Double>> testFeatures = new ArrayList<>();
        for (int i = 0; i < testList.get(0) ; i++) {
            ArrayList<Double> inputTemp = new ArrayList<>();
            testFeatures.add(inputTemp);
        }

        ArrayList<ArrayList<Double>> testOutputs = new ArrayList<>();
        for (int i = 0; i < testList.get(2) ; i++) {
            ArrayList<Double> outputTemp = new ArrayList<>();
            testOutputs.add(outputTemp);
        }


        for (int i = 4; i < testList.get(3) * (testList.get(0) + testList.get(2) ) ; i+= (testList.get(0) + testList.get(2) ) ) {

            int inputSizeTemp = 0;
            int outputSizeTemp = 0;
            while (inputSizeTemp != testList.get(0)){
                testFeatures.get(inputSizeTemp).add(testList.get(i + inputSizeTemp));
                inputSizeTemp++;
            }
            while (outputSizeTemp != testList.get(2)){
                testOutputs.get(outputSizeTemp).add(testList.get(i + outputSizeTemp));
                outputSizeTemp++;
            }

        }
        //end of splitting data

        //normalizing all the features and all the outputs into new arraylists

        //Mean
        ArrayList<Double> FeaturesMeans = ANN.mean(testFeatures);
        ArrayList<Double> OutputsMeans = ANN.mean(testOutputs);
        //std dev
        ArrayList<Double> FeaturesSTDs = ANN.StdDev(testFeatures , FeaturesMeans);
        ArrayList<Double> OutputsSTDs = ANN.StdDev(testOutputs , OutputsMeans);
        //Normalized Data (features , outputs)
        for (int i = 0; i < testFeatures.size() ; i++) {
            for (int j = 0; j < testFeatures.get(i).size() ; j++) {
                testFeatures.get(i).set(j , ( testFeatures.get(i).get(j) - FeaturesMeans.get(i) ) / FeaturesSTDs.get(i) );
            }
        }

        for (int i = 0; i < testOutputs.size() ; i++) {
            for (int j = 0; j < testOutputs.get(i).size() ; j++) {
                testOutputs.get(i).set(j , ( testOutputs.get(i).get(j) - OutputsMeans.get(i) ) / OutputsSTDs.get(i) );
            }
        }

        System.out.println("preprocessing data has been Accomplished");
        // establishing the Neural Network


        double inputNodes[] = new double[sizeOfInputs];
        double hiddenNodes[] = new double[sizeOfHiddens];
        double predictedOutputNodes[] = new double[sizeOfOutputs];
        double actualOutputNodes[] = new double[sizeOfOutputs];
        double MSE = 0;
        //testing the model using FF
        int row = 0;
        // examples to be used for training
        for (int i = 0; i < testList.get(3) ; i++) {
            //bias in inputs
            inputNodes[inputNodes.length-1] = 1;
            //feed forward
            MSE = ANN.FF( testFeatures,testOutputs, row, inputNodes, hiddenNodes, predictedOutputNodes,actualOutputNodes,hWeights, oWeights, MSE );
//                System.out.println("FF has been Accomplished");

//                System.out.println("BP has been Accomplished");
            Arrays.fill(hiddenNodes,0);
            Arrays.fill(predictedOutputNodes,0);
            row++;

        }
        System.out.println("Testing the model has been Accomplished");
        System.out.println("MSE for each example after testing them all =" +
                " MSE for all examples in the testing / number of examples" + "= " + MSE/testList.get(3));

    }
}
