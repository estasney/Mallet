package cc.mallet.topics;

import java.util.*;
import java.io.*;


import cc.mallet.extract.Field;
import cc.mallet.types.*;

import cc.mallet.util.Randoms;

import com.carrotsearch.hppc.ObjectDoubleHashMap;
import com.carrotsearch.hppc.IntIntHashMap;
import com.carrotsearch.hppc.cursors.IntIntCursor;
import org.apache.commons.lang3.StringUtils;

public class HierarchicalLDA implements Serializable {

    InstanceList instances;
    InstanceList testing;
    NCRPNode rootNode, node;

    int numLevels;
    int numDocuments;
    int numTypes;


    double[] alpha; // smoothing on topic distributions. A Document-Topic Prior
    double[] gamma; // "imaginary" customers at the next, as yet unused table
    double[] eta;   // smoothing on word distributions. A Topic-Word Prior
    double[] etaSum;

    int[][] levels; // indexed < doc, token >. Each document will have tokens assigned to a level
    NCRPNode[] documentLeaves; // currently selected path (ie leaf node) through the NCRP tree

    int totalNodes = 0;
    int[] levelTotalNodes;
    long[] levelTotalTokens;
    int iterationsRun = 0;

    String inputFile;
    String stateFile;
    String topicFile;
    String modelFile;

    Randoms random;

    int displayTopicsInterval = 50;
    int numWordsToDisplay = 10;
    int saveEvery = 0;

    public HierarchicalLDA() {
        alpha = new double[]{};
        gamma = new double[]{};
        eta = new double[]{};
        etaSum = new double[]{};
    }

    public void setAlpha(double[] alpha) {
        this.alpha = alpha;
    }

    public void setGamma(double[] gamma) {
        this.gamma = gamma;
    }

    public void setEta(double[] eta) {
        this.eta = eta;
    }

    public void setTopicDisplay(int interval, int words) {
        displayTopicsInterval = interval;
        numWordsToDisplay = words;
    }

    /**
     * This parameter determines whether the sampler outputs
     * shows progress by outputting a character after every iteration.
     */

    public void setSaveEvery(int value) {
        this.saveEvery = value;
    }

    public void setSaveState(String stateFileValue, String topicFileValue, String modelFileValue) {
        this.stateFile = stateFileValue;
        this.topicFile = topicFileValue;
        this.modelFile = modelFileValue;
    }

    public void setInputFile(String inputFile) {
        this.inputFile = inputFile;
    }

    public void initialize(InstanceList instances, InstanceList testing,
                           int numLevels, Randoms random) {
        this.instances = instances;
        this.testing = testing;
        this.numLevels = numLevels;
        this.levelTotalNodes = new int[numLevels];
        this.levelTotalTokens = new long[numLevels];
        this.random = random;
        this.etaSum = new double[numLevels];


        if (!(instances.get(0).getData() instanceof FeatureSequence)) {
            throw new IllegalArgumentException("Input must be a FeatureSequence, using the --feature-sequence option when importing data, for example");
        }

        numDocuments = instances.size();
        numTypes = instances.getDataAlphabet().size();

        for (int i = 0; i < this.numLevels; i++) {
            etaSum[i] = this.eta[i] * numTypes;
        }

        // Initialize a single path

        NCRPNode[] path = new NCRPNode[numLevels];

        rootNode = new NCRPNode(numTypes);

        levels = new int[numDocuments][];
        documentLeaves = new NCRPNode[numDocuments];

        // Initialize and fill the topic pointer arrays for
        //  every document. Set everything to the single path that
        //  we added earlier.
        for (int doc = 0; doc < numDocuments; doc++) {
            FeatureSequence fs = (FeatureSequence) instances.get(doc).getData();
            int seqLen = fs.getLength();

            // assign first step as rootnode
            path[0] = rootNode;
            rootNode.customers++;
            for (int level = 1; level < numLevels; level++) {
                path[level] = path[level - 1].select();  // TODO burn-in
                path[level].customers++;
            }
            node = path[numLevels - 1];

            // fill levels[doc_idx] with 0's with length of unique tokens
            levels[doc] = new int[seqLen];
            documentLeaves[doc] = node;

            for (int token = 0; token < seqLen; token++) {
                int type = fs.getIndexAtPosition(token);
                levels[doc][token] = random.nextInt(numLevels);  // TODO pass params?
                node = path[levels[doc][token]];
                node.totalTokens++;
                node.typeCounts[type]++;
            }
        }
    }

    public void countNodeLevels(NCRPNode node, int level) {
        levelTotalNodes[level] += 1;
        levelTotalTokens[level] += node.totalTokens;
        for (NCRPNode child : node.children) {
            countNodeLevels(child, child.level);
        }
    }

    public void countNodeLevels(NCRPNode node, boolean reset) {
        // node counts it children and recursively calls countNodeLevels on its children
        if (reset) {
            for (int i = 1; i < numLevels; i++) {
                levelTotalNodes[i] = 0;
                levelTotalTokens[i] = 0;
            }

        }

        for (NCRPNode child : node.children) {
            countNodeLevels(child, child.level);
        }
    }

    public String showLevelCounts(String prefix, int outerIter, int outerTotal, int innerIter, int innerTotal, double timingmean) {
        countNodeLevels(rootNode, true);
        StringBuffer progress = new StringBuffer();
        progress.append("Iter ").append(outerIter).append(" of ").append(outerTotal).append(" : ").append(prefix).append(innerIter + 1).append(" of ").append(innerTotal).append(" ms/iter : ")
                .append(timingmean).append(" ");

        progress.append("(");
        for (int level = 0; level < numLevels; level++) {
            progress.append(levelTotalNodes[level]);
            progress.append(":");
            progress.append(levelTotalTokens[level]);
            if (level + 1 != numLevels) {
                progress.append(", ");
            } else {
                progress.append(")");
            }
        }
        return progress.toString();
    }

    public void estimate(int numIterations) throws IOException {
        long startTime, runningTotal;
        double totalTimings = 0;
        double timingMean = 0;
        int lastLineLength = 0;
        int lineLength = 0;
        runningTotal = 0;

        for (int iteration = 1; iteration <= numIterations; iteration++) {
            for (int doc = 0; doc < numDocuments; doc++) {
                startTime = System.currentTimeMillis();
                samplePath(doc);
                runningTotal += (System.currentTimeMillis() - startTime);
                totalTimings += 1;

                if (doc % 50 == 0 || (doc + 1) == numDocuments) {

                    // update the mean timings
                    timingMean = runningTotal / totalTimings;
                    runningTotal = 0;
                    totalTimings = 0;
                    String progress = showLevelCounts("Sample Path : ", iteration, numIterations, doc, numDocuments, timingMean);
                    lineLength = progress.length();
                    System.out.print("\r");
                    if (lineLength < lastLineLength) {
                        System.out.print(StringUtils.repeat(" ", lastLineLength));
                        System.out.print("\r");
                    }
                    System.out.print(progress);
                    lastLineLength = lineLength;
                }
            }
            runningTotal = 0;
            totalTimings = 0;
            for (int doc = 0; doc < numDocuments; doc++) {
                startTime = System.currentTimeMillis();
                sampleTopics(doc);
                runningTotal += (System.currentTimeMillis() - startTime);
                totalTimings += 1;
                if (doc % 50 == 0 || (doc + 1) == numDocuments) {
                    timingMean = runningTotal / totalTimings;
                    runningTotal = 0;
                    totalTimings = 0;
                    String progress = showLevelCounts("Sample Topics : ", iteration, numIterations, doc, numDocuments, timingMean);
                    lineLength = progress.length();
                    System.out.print("\r");
                    if (lineLength < lastLineLength) {
                        System.out.print(StringUtils.repeat(" ", lastLineLength));
                        System.out.print("\r");
                    }
                    System.out.print(progress);
                    lastLineLength = lineLength;
                }
            }
            System.out.print("\n");
            countNodeLevels(rootNode, true);

            if (iteration % displayTopicsInterval == 0) {
                printNodes();
            }

            if (saveEvery > 0 & iteration > 0 & iteration % saveEvery == 0) {
                if (stateFile != null) {
                    printState(new PrintWriter(stateFile));
                }
                if (topicFile != null) {
                    printEdgeList(topicFile);
                }
                if (modelFile != null) {
                    File modelFileOut = new File(modelFile);
                    write(modelFileOut);
                }
                if (stateFile != null || topicFile != null || modelFile != null) {
                    System.out.println("Saved state");
                }

            }

        iterationsRun += 1;
        }
    }

    public void samplePath(int doc) {
        NCRPNode[] path = new NCRPNode[numLevels];
        NCRPNode node;
        int level;
        int token;
        int type;

        double levelEta;
        double levelEtaSum;

        node = documentLeaves[doc];
        for (level = numLevels - 1; level >= 0; level--) {
            path[level] = node;
            node = node.parent;
        }

        documentLeaves[doc].dropPath();

        ObjectDoubleHashMap<NCRPNode> nodeWeights =
                new ObjectDoubleHashMap<NCRPNode>();

        // Calculate p(c_m | c_{-m})
        calculateNCRP(nodeWeights, rootNode, 0.0);

        // Add weights for p(w_m | c, w_{-m}, z)

        // The path may have no further customers and therefore
        //  be unavailable, but it should still exist since we haven't
        //  reset documentLeaves[doc] yet...

        IntIntHashMap[] typeCounts = new IntIntHashMap[numLevels];

        int[] docLevels;

        for (level = 0; level < numLevels; level++) {
            typeCounts[level] = new IntIntHashMap();
        }

        docLevels = levels[doc];
        FeatureSequence fs = (FeatureSequence) instances.get(doc).getData();

        // Save the counts of every word at each level, and remove
        //  counts from the current path

        for (token = 0; token < docLevels.length; token++) {
            level = docLevels[token];
            type = fs.getIndexAtPosition(token);

            if (!typeCounts[level].containsKey(type)) {
                typeCounts[level].put(type, 1);
            } else {
                typeCounts[level].addTo(type, 1);
            }

            path[level].typeCounts[type]--;
            assert (path[level].typeCounts[type] >= 0);

            path[level].totalTokens--;
            assert (path[level].totalTokens >= 0);
        }

        // Calculate the weight for a new path at a given level.
        double[] newTopicWeights = new double[numLevels];
        for (level = 1; level < numLevels; level++) {  // Skip the root...
            int totalTokens = 0;
            levelEta = eta[level];
            levelEtaSum = etaSum[level];

            for (IntIntCursor keyVal : typeCounts[level]) {
                for (int i = 0; i < keyVal.value; i++) {
                    newTopicWeights[level] +=
                            Math.log((levelEta + i) / (levelEtaSum + totalTokens));
                    totalTokens++;
                }
            }

//			if (iteration > 1) { System.out.println(newTopicWeights[level]); }
        }

        calculateWordLikelihood(nodeWeights, rootNode, 0.0, typeCounts, newTopicWeights, 0);

        Object[] objectArray = nodeWeights.keys().toArray();
        NCRPNode[] nodes = Arrays.copyOf(objectArray, objectArray.length, NCRPNode[].class);
        double[] weights = new double[nodes.length];
        double sum = 0.0;
        double max = Double.NEGATIVE_INFINITY;

        // To avoid underflow, we're using log weights and normalizing the node weights so that
        //  the largest weight is always 1.
        for (int i = 0; i < nodes.length; i++) {
            if (nodeWeights.get(nodes[i]) > max) {
                max = nodeWeights.get(nodes[i]);
            }
        }

        for (int i = 0; i < nodes.length; i++) {
            weights[i] = Math.exp(nodeWeights.get(nodes[i]) - max);

            sum += weights[i];
        }

        int choice = random.nextDiscrete(weights, sum);
        node = nodes[choice];

        // If we have picked an internal node, we need to
        //  add a new path.
        if (!node.isLeaf()) {
            node = node.getNewLeaf();
        }

        node.addPath(); // increase customers up this node's parents
        documentLeaves[doc] = node;

        for (level = numLevels - 1; level >= 0; level--) {

            for (IntIntCursor keyVal : typeCounts[level]) {
                node.typeCounts[keyVal.key] += keyVal.value;
                node.totalTokens += keyVal.value;
            }

            node = node.parent;
        }
    }

    public void calculateNCRP(ObjectDoubleHashMap<NCRPNode> nodeWeights,
                              NCRPNode node, double weight) {
        double levelGamma;

        // Here gamma's effect is more pronounced for nodes where customers is very low. As customers grow, gamma's
        // effect diminishes

        // A higher gamma will decrease the weight assigned to a child


        for (NCRPNode child : node.children) {
            double innerWeight = Math.log((double) child.customers / (node.customers + gamma[node.level]));
            calculateNCRP(nodeWeights, child, innerWeight);
        }

        if (node.level + 1 == numLevels) {
            levelGamma = gamma[node.level];
        } else {
            levelGamma = gamma[(node.level + 1)];
        }

        nodeWeights.put(node, weight + Math.log(levelGamma / (node.customers + levelGamma)));
    }

    public void calculateWordLikelihood(ObjectDoubleHashMap<NCRPNode> nodeWeights,
                                        NCRPNode node, double weight,
                                        IntIntHashMap[] typeCounts, double[] newTopicWeights,
                                        int level) {

        // First calculate the likelihood of the words at this level, given
        //  this topic.
        double nodeWeight = 0.0;
        int totalTokens = 0;
        double levelEta, levelEtaSum;

        levelEta = eta[node.level];
        levelEtaSum = etaSum[node.level];


        for (IntIntCursor keyVal : typeCounts[level]) {
            for (int i = 0; i < keyVal.value; i++) {
                nodeWeight +=
                        Math.log((levelEta + node.typeCounts[keyVal.key] + i) /
                                (levelEtaSum + node.totalTokens + totalTokens));
                totalTokens++;

            }
        }

        // Propagate that weight to the child nodes

        for (NCRPNode child : node.children) {
            calculateWordLikelihood(nodeWeights, child, weight + nodeWeight,
                    typeCounts, newTopicWeights, level + 1);
        }

        // Finally, if this is an internal node, add the weight of
        //  a new path

        level++;
        while (level < numLevels) {
            nodeWeight += newTopicWeights[level];
            level++;
        }

        nodeWeights.addTo(node, nodeWeight);

    }

    /**
     * Propagate a topic weight to a node and all its children.
     * weight is assumed to be a log.
     */
    public void propagateTopicWeight(ObjectDoubleHashMap<NCRPNode> nodeWeights,
                                     NCRPNode node, double weight) {
        if (!nodeWeights.containsKey(node)) {
            // calculating the NCRP prior proceeds from the
            //  root down (ie following child links),
            //  but adding the word-topic weights comes from
            //  the bottom up, following parent links and then
            //  child links. It's possible that the leaf node may have
            //  been removed just prior to this round, so the current
            //  node may not have an NCRP weight. If so, it's not
            //  going to be sampled anyway, so ditch it.
            return;
        }

        for (NCRPNode child : node.children) {
            propagateTopicWeight(nodeWeights, child, weight);
        }

        nodeWeights.addTo(node, weight);
    }

    public void sampleTopics(int doc) {
        FeatureSequence fs = (FeatureSequence) instances.get(doc).getData();
        int seqLen = fs.getLength();
        int[] docLevels = levels[doc];
        NCRPNode[] path = new NCRPNode[numLevels];
        NCRPNode node;
        int[] levelCounts = new int[numLevels];
        int type, token, level;
        double sum;

        // Get the leaf and copy it to path
        node = documentLeaves[doc];
        for (level = numLevels - 1; level >= 0; level--) {
            path[level] = node;
            node = node.parent;
        }

        double[] levelWeights = new double[numLevels];

        // Initialize level counts
        for (token = 0; token < seqLen; token++) {
            levelCounts[docLevels[token]]++;
        }

        for (token = 0; token < seqLen; token++) {
            type = fs.getIndexAtPosition(token);

            levelCounts[docLevels[token]]--;
            node = path[docLevels[token]];  // get the node at level 0, level 1, ... that the token is assigned to
            node.typeCounts[type]--;
            node.totalTokens--;


            sum = 0.0;

            // Pick existing or new level for the token based on below calc
            for (level = 0; level < numLevels; level++) {
                levelWeights[level] =
                        (alpha[level] + levelCounts[level]) *
                                (eta[level] + path[level].typeCounts[type]) /
                                (etaSum[level] + path[level].totalTokens);
                sum += levelWeights[level];
            }
            level = random.nextDiscrete(levelWeights, sum);

            docLevels[token] = level;
            levelCounts[docLevels[token]]++;
            node = path[level];
            node.typeCounts[type]++;
            node.totalTokens++;
        }
    }

    /**
     * Writes the current sampling state to the file specified in <code>stateFile</code>.
     */
    public void printState() throws IOException, FileNotFoundException {
        printState(new PrintWriter(new BufferedWriter(new FileWriter(stateFile))));
    }

    /**
     * Write a csv file describing the current sampling state.
     */
    public void printState(PrintWriter out) throws IOException {
        int doc = 0;
        StringBuffer header = new StringBuffer();
        int headerLevel;
        for (headerLevel = 0; headerLevel < numLevels; headerLevel++) {
            header.append("Level_").append(headerLevel).append("_ID").append(",");
            header.append("Level_").append(headerLevel).append("_Customers").append(",");
            header.append("Level_").append(headerLevel).append("_NTokens").append(",");
        }

        header.append("Node_ID,Node_Customers,Node_NTokens,Token,Token_Level,Token_Weight");
        out.println(header);

        // instances are documents, get all tokens present in all documents
        Alphabet alphabet = instances.getDataAlphabet();

        // for document in documents...
        for (Instance instance : instances) {

            FeatureSequence fs = (FeatureSequence) instance.getData();
            int seqLen = fs.getLength();
            // get array of levels that each token in document is assigned
            int[] docLevels = levels[doc];
            NCRPNode node;
            NCRPNode childNode;
            int type, token, level;
            double tokenWeight;

            StringBuffer path = new StringBuffer();

            // Start with the leaf, and build a string describing the path for this doc
            // documentLeaves are all at lowest level
            node = documentLeaves[doc];

            // this is the lowest level node
            childNode = documentLeaves[doc];

            // this describes the hierarchy of the nodes
            for (level = numLevels - 1; level >= 0; level--) {
                path.append(node.nodeID).append(",").append(node.customers).append(",").append(node.totalTokens).append(",");
                node = node.parent;
            }

            path.append(childNode.nodeID).append(",").append(childNode.customers).append(",").append(childNode.totalTokens).append(",");

            List<Double> nodeWeights = childNode.getTopWeights(seqLen);
            String pathOut = path.toString();

            for (token = 0; token < seqLen; token++) {
                type = fs.getIndexAtPosition(token);
                level = docLevels[token];
                tokenWeight = nodeWeights.get(token);
                // The "" just tells java we're not trying to add a string and an int
                Object alphaObject = alphabet.lookupObject(type);
                String alphaString = alphaObject.toString();
                String tokenDataSafe = this.escapeSpecialCharacters(alphaString);
                String tokenData = pathOut + "" + type + "," + tokenDataSafe + "," + level + "," + tokenWeight;

                out.println(tokenData);
            }
            doc++;
        }
        out.close();
    }

    public String escapeSpecialCharacters(String data) {
        String escapedData = data.replaceAll("\\R", " ");
        if (data.contains(",") || data.contains("\"") || data.contains("'")) {
            data = data.replace("\"", "\"\"");
            escapedData = "\"" + data + "\"";
        }
        return escapedData;
    }

    public void printEdgeList(String fp) throws IOException {
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(fp)));
        out.write("parent,child,child_id,type,child_weight,parent_weight,parent_level" + "\n");

        // Create token 2 id mapping
        Map<String, Integer> token2Id = new HashMap<String, Integer>();

        Alphabet alphabet = instances.getDataAlphabet();
        ArrayList alphabetArray = alphabet.entries;

        for (int i = 0; i < alphabetArray.size(); i++) {

            // Prevent token id collisions with nodes
            String token = (String) alphabetArray.get(i);
            String tokenSafe = this.escapeSpecialCharacters(token);
            int tokenId = i + this.totalNodes;
            token2Id.put(tokenSafe, tokenId);
        }

        printNodeEdge(rootNode, out, token2Id);
        out.close();
    }

    public void printTopicNodes(String fp) throws IOException {
        PrintWriter out1 = new PrintWriter(new BufferedWriter(new FileWriter(fp)));
        out1.write("Node_ID,Node_Level,Word,Weight" + "\n");
        printNode(rootNode, 0, true, true, out1);
        out1.close();
    }

    public void printNodeEdge(NCRPNode node, PrintWriter out, Map<String, Integer> token2id) {
        // alphabet is used to retrieve token ids
        // get all words with a weight greater than 0
        int nnz = Arrays.stream(node.typeCounts).filter(x -> x > 0).toArray().length;

        if (nnz >= numWordsToDisplay) {
            String topwords = node.getTopWords(nnz, true);
            int nodeLevel = node.level;
            String[] tempArray;

            String delimiter = " ";
            tempArray = topwords.split(delimiter);
            /*
             * Print this nodes top words and node data
             * "parent,child,child_id,child_type,child_weight,parent_weight,parent_level"
             * */
            for (String s : tempArray) {

                String[] lineTempArray;
                lineTempArray = s.split(":");
                String tokenSafe = this.escapeSpecialCharacters(lineTempArray[0]);
                int tokenId = token2id.get(tokenSafe);
                String csvString = node.nodeID + "," + tokenSafe + "," + tokenId + ",word," + lineTempArray[1] + "," + node.customers + "," + nodeLevel;
                if (out == null) {
                    System.out.println(csvString);
                } else {
                    out.println(csvString);
                }

            }
        }


		/*
		Print this nodes children
		parent, child, type, weight
		 */
        for (int i = 0; i < node.children.size(); i++) {
            int childId = node.children.get(i).nodeID;
            int childCustomers = node.children.get(i).customers;
            int childLevel = node.children.get(i).level;
            String csvString = node.nodeID + "," + childId + "," + childId + ",node," + childCustomers + "," + node.customers + "," + node.level;
            if (out == null) {
                System.out.println(csvString);
            } else {
                out.println(csvString);
            }

        }

        for (NCRPNode child : node.children) {
            printNodeEdge(child, out, token2id);
        }
    }

    public void printNodes() {
        printNode(rootNode, 0, false, false, null);
    }

    public void printNodes(boolean withWeight) {
        printNode(rootNode, 0, withWeight, false, null);
    }

    public void printNode(NCRPNode node, int indent, boolean withWeight, boolean csvFormat, PrintWriter out) {

        String sepChar;
        if (!csvFormat) {
            sepChar = "  ";
        } else {
            sepChar = ",";
        }

        StringBuffer path = new StringBuffer();
        if (!csvFormat) {
            for (int i = 0; i < indent; i++) {
                path.append(sepChar);
            }
        }


        if (!csvFormat) {
            path.append("n_tokens: ").append(node.totalTokens).append("/n_customers: ").append(node.customers).append(" ");
            path.append(node.getTopWords(numWordsToDisplay, withWeight));
        } else {

            // get an arraylist of Word, Weight
            // make an edgelist
            String topwords = node.getTopWords(numWordsToDisplay, withWeight);
            String[] tempArray;

            String delimiter = " ";
            tempArray = topwords.split(delimiter);
            for (String s : tempArray) {

                String[] lineTempArray;
                lineTempArray = s.split(":");
                String tokenSafe = this.escapeSpecialCharacters(lineTempArray[0]);
                String csvString = node.nodeID + "," + node.level + "," + tokenSafe + "," + lineTempArray[1];
                if (out == null) {
                    System.out.println(csvString);
                } else {
                    out.println(csvString);
                }

            }
        }


        if (out == null) {
            System.out.println(path);
        } else {
            if (!csvFormat) {
                out.println(path);
            }

        }


        for (NCRPNode child : node.children) {
            printNode(child, indent + 1, withWeight, csvFormat, out);
        }
    }

    /**
     * For use with empirical likelihood evaluation:
     * sample a path through the tree, then sample a multinomial over
     * topics in that path, then return a weighted sum of words.
     */
    public double empiricalLikelihood(int numSamples, InstanceList testing) {
        NCRPNode[] path = new NCRPNode[numLevels];
        NCRPNode node;
        double weight;
        double levelEta, levelEtaSum;


        path[0] = rootNode;

        FeatureSequence fs;
        int sample, level, type, token, doc, seqLen, i;

        Dirichlet dirichlet = new Dirichlet(numLevels, alpha);
        double[] levelWeights;
        double[] multinomial = new double[numTypes];

        double[][] likelihoods = new double[testing.size()][numSamples];

        for (sample = 0; sample < numSamples; sample++) {
            Arrays.fill(multinomial, 0.0);

            for (level = 1; level < numLevels; level++) {
                path[level] = path[level - 1].selectExisting();
            }

            levelWeights = dirichlet.nextDistribution();

            for (type = 0; type < numTypes; type++) {
                for (level = 0; level < numLevels; level++) {
                    levelEta = eta[level];
                    levelEtaSum = etaSum[level];
                    node = path[level];
                    multinomial[type] +=
                            levelWeights[level] *
                                    (levelEta + node.typeCounts[type]) /
                                    (levelEtaSum + node.totalTokens);
                }

            }

            for (type = 0; type < numTypes; type++) {
                multinomial[type] = Math.log(multinomial[type]);
            }

            for (doc = 0; doc < testing.size(); doc++) {
                fs = (FeatureSequence) testing.get(doc).getData();
                seqLen = fs.getLength();

                for (token = 0; token < seqLen; token++) {
                    type = fs.getIndexAtPosition(token);
                    likelihoods[doc][sample] += multinomial[type];
                }
            }
        }

        double averageLogLikelihood = 0.0;
        double logNumSamples = Math.log(numSamples);
        for (doc = 0; doc < testing.size(); doc++) {
            double max = Double.NEGATIVE_INFINITY;
            for (sample = 0; sample < numSamples; sample++) {
                if (likelihoods[doc][sample] > max) {
                    max = likelihoods[doc][sample];
                }
            }

            double sum = 0.0;
            for (sample = 0; sample < numSamples; sample++) {
                sum += Math.exp(likelihoods[doc][sample] - max);
            }

            averageLogLikelihood += Math.log(sum) + max - logNumSamples;
        }

        return averageLogLikelihood;
    }

    public void write(File serializedModelFile) {
        try {
            ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(serializedModelFile));
            oos.writeObject(this);
            oos.close();
        } catch (IOException e) {
            System.err.println("Problem serializing HierarchicalLDA to file " +
                    serializedModelFile + ": " + e);
        }
    }

    public static String readParams(File f) throws Exception{
        HierarchicalLDA topicModel = read(f);

        StringBuilder output = new StringBuilder();

        String[] paramNamesStr = {"inputFile", "stateFile", "topicFile", "numLevels", "iterationsRun"};
        String[] paramNamesArr = {"alpha", "gamma", "eta"};

        for (String p : paramNamesStr) {
            String value = (String) topicModel.getClass().getField(p).get(topicModel);
            output.append(p).append(" : ");
            output.append(value).append("\n");
        }

        for (String p : paramNamesArr) {
            String value = java.util.Arrays.toString((double[]) topicModel.getClass().getField(p).get(topicModel));
            output.append(p).append(" : ");
            output.append(value).append("\n");
        }

        return output.toString();






    }

    public static HierarchicalLDA read(File f) throws Exception {

        HierarchicalLDA topicModel;

        ObjectInputStream ois = new ObjectInputStream(new FileInputStream(f));
        topicModel = (HierarchicalLDA) ois.readObject();
        ois.close();

        return topicModel;
    }

    /**
     * This method is primarily for testing purposes. The {@link cc.mallet.topics.tui.HierarchicalLDATUI}
     * class has a more flexible interface for command-line use.
     */
    public static void main(String[] args) {
        try {
            InstanceList instances = InstanceList.load(new File(args[0]));
            InstanceList testing = InstanceList.load(new File(args[1]));

            HierarchicalLDA sampler = new HierarchicalLDA();
            sampler.initialize(instances, testing, 5, new Randoms());
            sampler.estimate(250);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    class NCRPNode implements Serializable {
        int customers;
        ArrayList<NCRPNode> children;
        NCRPNode parent;
        int level;

        int totalTokens;
        int[] typeCounts;

        public int nodeID;

        public NCRPNode(NCRPNode parent, int dimensions, int level) {
            customers = 0;
            this.parent = parent;
            children = new ArrayList<NCRPNode>();
            this.level = level;

//			System.out.println("new node at level " + level);

            totalTokens = 0;
            typeCounts = new int[dimensions];

            nodeID = totalNodes;
            totalNodes++;
            levelTotalNodes[level]++;
        }

        public NCRPNode(int dimensions) {
            this(null, dimensions, 0);
        }

        public NCRPNode addChild() {
            NCRPNode node = new NCRPNode(this, typeCounts.length, level + 1);
            children.add(node);
            return node;
        }

        public boolean isLeaf() {
            return level == numLevels - 1;
        }

        public NCRPNode getNewLeaf() {
            NCRPNode node = this;
            for (int l = level; l < numLevels - 1; l++) {
                node = node.addChild();
            }
            return node;
        }

        public void dropPath() {
            NCRPNode node = this;
            // remove 1 customer from node ... does it survive?
            node.customers--;
            if (node.customers == 0 || node.totalTokens == 0) {
//				System.out.println("Node removed at level " + node.level);
                node.parent.remove(node);
            }
            // do the same for parent nodes
            for (int l = 1; l < numLevels; l++) {
                node = node.parent;
                node.customers--;
                if (node.customers == 0 || node.totalTokens == 0) {
//					System.out.println("Node removed at level " + node.level);
                    node.parent.remove(node);
                }
            }
        }

        public void remove(NCRPNode node) {
            children.remove(node);
        }

        public void addPath() {
            NCRPNode node = this;
            node.customers++;
            for (int l = 1; l < numLevels; l++) {
                node = node.parent;
                node.customers++;
            }
        }

        public NCRPNode selectExisting() {
            double[] weights = new double[children.size()];
            double weightsSum = 0;

            int i = 0;
            for (NCRPNode child : children) {
                weights[i] = (double) child.customers / (gamma[child.level] + customers);
                i++;
            }

            for (double weight : weights) {
                weightsSum += weight;
            }

            int choice = random.nextDiscrete(weights, weightsSum);
            return children.get(choice);
        }

        public NCRPNode select() {
            // creating an array of weights
            double[] weights = new double[children.size() + 1];
            double weightsSum = 0;  // Calculated for normalization


            int i = 1;
            for (NCRPNode child : this.children) {
                weights[i] = (double) child.customers / (gamma[child.level] + customers);
                i++;
            }

//            if (this.level + 1 == numLevels) {
//                levelGamma = gamma[this.level];
//            } else {
//                levelGamma = gamma[(this.level + 1)];
//            }

            weights[0] = gamma[this.level] / (gamma[this.level] + customers);

            for (double weight : weights) {
                weightsSum += weight;
            }

            int choice = random.nextDiscrete(weights, weightsSum);
            if (choice == 0) {
                return (addChild());
            } else {
                return children.get(choice - 1);
            }
        }

        public String getTopWords(int numWords, boolean withWeight) {
            IDSorter[] sortedTypes = new IDSorter[numTypes];

            for (int type = 0; type < numTypes; type++) {
                sortedTypes[type] = new IDSorter(type, typeCounts[type]);
            }
            Arrays.sort(sortedTypes);

            Alphabet alphabet = instances.getDataAlphabet();
            StringBuffer out = new StringBuffer();
            for (int i = 0; i < numWords; i++) {
                if (withWeight) {
                    out.append(alphabet.lookupObject(sortedTypes[i].getID()) + ":" + sortedTypes[i].getWeight() + " ");
                } else
                    out.append(alphabet.lookupObject(sortedTypes[i].getID()) + " ");
            }
            return out.toString();
        }

        public List<Double> getTopWeights(int numWords) {
            List<Double> weights = new ArrayList<Double>();
            IDSorter[] sortedTypes = new IDSorter[numTypes];

            for (int type = 0; type < numTypes; type++) {
                sortedTypes[type] = new IDSorter(type, typeCounts[type]);
            }
            Arrays.sort(sortedTypes);

            Alphabet alphabet = instances.getDataAlphabet();
            for (int i = 0; i < numWords; i++) {
                weights.add(sortedTypes[i].getWeight());
            }
            return weights;
        }

    }

}