package cc.mallet.topics.tui;

import cc.mallet.util.CommandOption;
import cc.mallet.util.Randoms;
import cc.mallet.types.InstanceList;
import cc.mallet.topics.HierarchicalLDA;

import java.io.*;
import java.util.Arrays;

public class HierarchicalLDATUI {
	
	static CommandOption.String inputFile = new CommandOption.String
		(HierarchicalLDATUI.class, "input", "FILENAME", true, null,
		 "The filename from which to read the list of training instances.  Use - for stdin.  " +
		 "The instances must be FeatureSequence or FeatureSequenceWithBigrams, not FeatureVector", null);
	
	static CommandOption.String testingFile = new CommandOption.String
		(HierarchicalLDATUI.class, "testing", "FILENAME", true, null,
		 "The filename from which to read the list of instances for held-out likelihood calculation.  Use - for stdin.  " +
		 "The instances must be FeatureSequence or FeatureSequenceWithBigrams, not FeatureVector", null);
	
	static CommandOption.String stateFile = new CommandOption.String
		(HierarchicalLDATUI.class, "output-state", "FILENAME", true, null,
		 "The filename in which to write the Gibbs sampling state after at the end of the iterations.  " +
		 "By default this is null, indicating that no file will be written.", null);

	static CommandOption.String outputModelFilename = new CommandOption.String(HierarchicalLDATUI.class, "output-model", "FILENAME", true, null,
			"The filename in which to write the binary topic model at the end of the iterations.  " +
					"By default this is null, indicating that no file will be written.", null);


	static CommandOption.String topicNodeFile = new CommandOption.String
			(HierarchicalLDATUI.class, "output-topics", "FILENAME", true, null,
					"Write printNode to file after training", null);
	
	static CommandOption.Integer randomSeed = new CommandOption.Integer
		(HierarchicalLDATUI.class, "random-seed", "INTEGER", true, 0,
		 "The random seed for the Gibbs sampler.  Default is 0, which will use the clock.", null);
	
	static CommandOption.Integer numIterations = new CommandOption.Integer
	  	(HierarchicalLDATUI.class, "num-iterations", "INTEGER", true, 1000,
		 "The number of iterations of Gibbs sampling.", null);

	static CommandOption.Boolean showProgress = new CommandOption.Boolean
		(HierarchicalLDATUI.class, "show-progress", "BOOLEAN", false, true,
		 "If true, print a character to standard output after every sampling iteration.", null);

	static CommandOption.Integer showTopicsInterval = new CommandOption.Integer
		(HierarchicalLDATUI.class, "show-topics-interval", "INTEGER", true, 50,
		 "The number of iterations between printing a brief summary of the topics so far.", null);

	static CommandOption.Integer showNodesInterval = new CommandOption.Integer
			(HierarchicalLDATUI.class, "node-count-interval", "INTEGER", true, 1,
					"The number of iterations between printing a count of nodes at each level.", null);

	static CommandOption.Integer topWords = new CommandOption.Integer
		(HierarchicalLDATUI.class, "num-top-words", "INTEGER", true, 20,
		 "The number of most probable words to print for each topic after model estimation.", null);

	static CommandOption.Integer numLevels = new CommandOption.Integer
		(HierarchicalLDATUI.class, "num-levels", "INTEGER", true, 3,
		 "The number of levels in the tree.", null);

	static CommandOption.DoubleArray alpha = new CommandOption.DoubleArray
		(HierarchicalLDATUI.class, "alpha", "DECIMAL,[DECIMAL,...]", true, new double[] {1, 1, 0.1},
		 "Alpha parameter: smoothing over level distributions.  "+
				"For example --alpha 10,10,10", null);

	static CommandOption.DoubleArray gamma = new CommandOption.DoubleArray
		(HierarchicalLDATUI.class, "gamma", "DECIMAL,[DECIMAL,...]", true, new double[] {1.0, 1.0, 0.1},
		 "Gamma parameter: CRP smoothing parameter; number of imaginary customers at next, as yet unused table   "+
				"For example --gamma 1,1,1", null);

	static CommandOption.DoubleArray eta = new CommandOption.DoubleArray
		(HierarchicalLDATUI.class, "eta", "DECIMAL,[DECIMAL,...]", true, new double[] {1, 1, 0.1},
		 "Eta parameter: smoothing over topic-word distributions", null);

	public static void main (String[] args) throws java.io.IOException {

		// Process the command-line options
		CommandOption.setSummary (HierarchicalLDATUI.class,
								  "Hierarchical LDA with a fixed tree depth.");

		CommandOption.process (HierarchicalLDATUI.class, args);



		// Load instance lists

		if (inputFile.value() == null) {
			System.err.println("Input instance list is required, use --input option");
			System.exit(1);
		}

		// Check that gamma, alpha and eta all have length equal to num_levels
		int argLength, nLevels;
		nLevels = numLevels.value();

		if (gamma.value().length != nLevels) {
			argLength = gamma.value().length;
			System.err.println("Gamma parameters are of length: " + argLength + " but numLevels is of length: " + nLevels);
			System.exit(1);
		}

		if (eta.value().length != nLevels) {
			argLength = eta.value().length;
			System.err.println("Eta parameters are of length: " + argLength + " but numLevels is of length: " + nLevels);
			System.exit(1);
		}

		if (alpha.value().length != nLevels) {
			argLength = alpha.value().length;
			System.err.println("Alpha parameters are of length: " + argLength + " but numLevels is of length: " + nLevels);
			System.exit(1);
		}

		InstanceList instances = InstanceList.load(new File(inputFile.value()));
		InstanceList testing = null;
		if (testingFile.value() != null) {
			testing = InstanceList.load(new File(testingFile.value()));
		}
	
		HierarchicalLDA hlda = new HierarchicalLDA();
		
		// Set hyperparameters

		hlda.setAlpha(alpha.value());
		hlda.setGamma(gamma.value());
		hlda.setEta(eta.value());
		
		// Display preferences

		hlda.setTopicDisplay(showTopicsInterval.value(), topWords.value(), showNodesInterval.value());
		hlda.setProgressDisplay(showProgress.value());

		// Initialize random number generator

		Randoms random = null;
		if (randomSeed.value() == 0) {
			random = new Randoms();
		}
		else {
			random = new Randoms(randomSeed.value());
		}

		// Initialize and start the sampler

		hlda.initialize(instances, testing, numLevels.value(), random);
		hlda.estimate(numIterations.value());
		
		// Output results

		if (stateFile.value() != null) {
			hlda.printState(new PrintWriter(stateFile.value()));
		}

		if (topicNodeFile.value() != null) {
//			hlda.printTopicNodes(topicNodeFile.value());
			hlda.printEdgeList(topicNodeFile.value());
		}

		if (outputModelFilename.value != null) {

			try {

				ObjectOutputStream oos =
						new ObjectOutputStream (new FileOutputStream (outputModelFilename.value));
				oos.writeObject (hlda);
				oos.close();

			} catch (Exception e) {
				System.out.println("Couldn't write topic model to filename " + outputModelFilename.value);
			}
		}

		if (testing != null) {
			double empiricalLikelihood = hlda.empiricalLikelihood(1000, testing);
			System.out.println("Empirical likelihood: " + empiricalLikelihood);
		}
		
	}
}
