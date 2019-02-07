# Gamma Process Poisson Factorization

## Overview

Gamma Process Poisson Factorization (GPPF) is a Scala package for network and topic modeling. It implements the Poisson factorization methods described in [1], which come in three varieties: network-only modeling (N-GPPF), corpus-only modeling (C-GPPF), and joint modeling (J-GPPF). Installation instructions follow, as well as file formats and a tutorial on joint modeling that uses bag-of-words bill summaries and a voting network derived from U.S. Senate voting records.

## Setup

We recommend using a Linux environment for GPPF. Both Java and Apache Maven must be installed. To build the project, run:

    mvn package

To train a model, first set the desired parameters in `poisson.config`, as described in the Configuration section. Then run:

    java -jar GPPF/target/GPPF.jar poisson.config

Optionally, python can be installed for running the included visualization functions. These require pyLDAvis (for LDAvis display) and NetworkX (for GML output).

To visualize the trained model with LDAvis [2], install pyLDAvis and clone the LDAvis github project into the directory containing the GPPF repository. If LDAvis has already been installed, update the path in `generate_LDAvis_results.py` to point to `LDAvis/inst/htmljs`. The following files will be copied to the visualization output:

    LDAvis/inst/htmljs/d3.v3.js
    LDAvis/inst/htmljs/lda.css
    LDAvis/inst/htmljs/ldavis.js

## Configuration

All parameters are passed through `poisson.config`, with variables as follows. The configuration file can be passed by command line to `GPPF.jar` and all python visualization functions, so that multiple configuration files can be created for different datasets:

    # Input Filenames (known data entries)
    NETWORK_TRAIN="SampleData/senate109/network.txt"        # required by N-GPPF and J-GPPF (ignored by C-GPPF)
    CORPUS_TRAIN="SampleData/senate109/corpus.txt"          # required by C-GPPF and J-GPPF (ignored by N-GPPF)
    AUTHORS_TRAIN="SampleData/senate109/authors.txt"        # required by J-GPPF only (ignored by C-GPPF and N-GPPF)
 
    # Input Filenames (unknown data entries)
    NETWORK_HELDOUT=""      # optional network heldout entries; set to empty string ("") to ignore
    CORPUS_HELDOUT=""       # optional corpus heldout entries; set to empty string ("") to ignore
    
    # Output Directory
    OUT_DIR="SampleData/senate109/results"       # GPPF model and the visualization functions will save results here
    
    # Parameters
    NETWORK_TOPICS="20"     # maximum number of latent factors for B
    CORPUS_TOPICS="20"      # maximum number of latent factors for Y
    BURNIN_ITER="1000"      # number of burnin iterations
    COLLECT_ITER="500"      # number of collection iterations (for expected values)
    OUTPUT_ITER="100"       # save latent samples every OUTPUT_ITER iterations
    COUNT_FLAG="1"          # 1: count-valued network, 0: binary network (ignored by C-GPPF)
    EPSILON="1.0"           # interaction between B and Y (> 0.0) (ignored by C-GPPF and N-GPPF)
    
    # Commands
    RUN_MODEL="jgppf"       # jgppf: run the joint model, ngppf: run the network-only model, cgppf: run the corpus-only model
    GENERATE_SAMPLES="0"    # 1: generate B and/or Y samples from the trained model, 0: don't generate samples
    
    # Display (optional)
    NETWORK_DICTIONARY="SampleData/senate109/network_dictionary.txt"        # author index mapping
    CORPUS_DICTIONARY="SampleData/senate109/corpus_dictionary.txt"          # word index mapping

## Input File Formats

The input files are tab-delimited text files that list nonzero entries.

### NETWORK\_TRAIN and NETWORK\_HELDOUT

The network files contain a header line and a list of edges, as follows. *AuthorN*\* and *AuthorM*\* are IDs in the network dictionary:

    NumberOfAuthors <tab> NumberOfEntries
    AuthorN1 <tab> AuthorM1 <tab> Count1
    AuthorN2 <tab> AuthorM2 <tab> Count2
    AuthorN3 <tab> AuthorM3 <tab> Count3
    ...

### CORPUS\_TRAIN and CORPUS\_HELDOUT

The corpus files contain a header line and a list of word counts, as follows. *Document*\* is the document index and *Word*\* is the word ID in the corpus dictionary:

    NumberOfDocuments <tab> NumberOfUniqueWords <tab> NumberOfEntries
    DocumentA <tab> WordA1 <tab> CountA1
    DocumentA <tab> WordA2 <tab> CountA2
    DocumentA <tab> WordA3 <tab> CountA3
    ...
    DocumentB <tab> WordB1 <tab> CountB1
    DocumentB <tab> WordB2 <tab> CountB2
    DocumentB <tab> WordB3 <tab> CountB3
    ...

### AUTHOR\_TRAIN

The author file is simply a list of authors and documents, as follows:

    AuthorA <tab> DocumentA1
    AuthorA <tab> DocumentA2
    AuthorA <tab> DocumentA3
    ...
    AuthorB <tab> DocumentB1
    AuthorB <tab> DocumentB2
    AuthorB <tab> DocumentB3
    ...

### CORPUS\_DICTIONARY and NETWORK\_DICTIONARY

The dictionaries are a list of word IDs and their corresponding text values. For the corpus the text is generally a dictionary word, and for the network the text is generally an author's name. *TotalCount*\* is optional and refers to the sum of the word's occurrences in the corpus.

    Word1Id <tab> Word1Text <tab> TotalCount1
    Word2Id <tab> Word2Text <tab> TotalCount2
    Word3Id <tab> Word3Text <tab> TotalCount3
    ...

## Output File Formats

GPPF generates the following output files:

| Filename    | Description |
| :---------- | :---------- |
| rkB.txt     | Overall network factor strengths. Each line is an entry in the rkB vector |
| phink.txt   | Topic-author network factor strengths. Each line is a tab-delimited column in the phi matrix (KB lines with N entries each) |
| psiwk.txt   | Topic-word network factor strengths. Each line is a tab-delimited column in the psi matrix (KB lines with V entries each) |
| rkY.txt     | Overall corpus factor strengths. Each line is an entry in the rkY vector |
| thetadk.txt | Topic-document corpus factor strengths. Each line is a tab-delimited column in the theta matrix (KY lines with D entries each) |
| betawk.txt  | Topic-word corpus factor strengths. Each line is a tab-delimited column in the beta matrix (KY lines with V entries each) |

## Tutorial

### Senate Dataset

Included with the code is a sample dataset that consists of Senate bill summaries and a voting network from the 109th U.S. Congress. The corpus is a bag-of-words representation of the bill summaries. We removed stop words, performed stemming and lemmatization (leaving only the base forms in the dictionary), and removed common and uncommon words. The network consists of voting parity between senators; i.e., the count values between pairs of senators are proportional to the number of times the pair agreed on a vote, whether yea or nay. We scaled the counts from [0,20] to improve the model fit given default hyperparameters. The "authors" file corresponds to the senators who voted yea for each bill.

### Modeling

To train a model with the Senate dataset, run `java -jar GPPF/target/GPPF.jar poisson.config` with the default configuration settings. The joint model typically takes several hours to fit the Senate dataset on a standard desktop machine. The network-only and corpus-only models run much faster, typically in a few seconds.

### Results and Visualization

Modeling results will be stored in `SampleData/Senate109/results`. To print the topics to command line, run:

    python display_topics.py poisson.config

This typically shows corpus topics relating to immigration, energy, healthcare, defense, and transportation, as well as network groups corresponding to the political parties.

To view the results with LDAvis, run:

    python generate_LDAvis_results.py poisson.config

Finally, to generate a GML file, which can be viewed with tools such as Gephi, run:

    python generate_GML_results.py poisson.config

## Advanced Use

### Hyperparameters

Hyperparameters can be set in the scala files: `NGPPFmodel.scala`, `CGPPFmodel.scala`, and `JGPPFmodel.scala`.

### Heldout Links

Links in the network or corpus can be heldout with the NETWORK\_HELDOUT and CORPUS\_HELDOUT files. These entries are then removed from the Gibbs update equations (i.e. they are considered to be "unknown" values, not zero values). One reason to use this functionality is to predict heldout links with the fitted model, for evaluation purposes. Another reason is if the links truly are "unknown", i.e. they are missing from the dataset.

### Epsilon

Set EPSILON in the configuration file to adjust the degree of joint modeling. Higher values will encourage greater interactions between the network and corpus. Adjusting EPSILON can also help if the count values in the network and corpus are scaled differently. We do not recommend setting EPSILON close to zero, as the J-GPPF Gibbs equations are designed for joint modeling. For disjoint modeling, run N-GPPF and C-GPPF separately.

### Samples Options

The Gibbs samples from burn-in and collection iterations can be viewed with `display_gibbs_samples.py`, with a resolution determined by OUTPUT\_ITER.

Synthetic samples can be generated from the fitted model with the GENERATE\_SAMPLES parameter. This will save a synthetic dataset that is statistically similar to the training data.

## References

[1] Acharya, Ayan, et al. "Gamma process poisson factorization for joint modeling of network and documents." *Joint European Conference on Machine Learning and Knowledge Discovery in Databases.* Springer, Cham, 2015.

[2] Sievert, Carson, and Kenneth Shirley. "LDAvis: A method for visualizing and interpreting topics." *Proceedings of the workshop on interactive language learning, visualization, and interfaces.* 2014.
