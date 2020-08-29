CogAlg
======

Full introduction: <www.cognitivealgorithm.info>

Intelligence is a general cognitive ability, ultimately the ability to predict. That includes planning, which technically is a self-prediction. Any prediction is interactive projection of known patterns, hence the first step must be pattern discovery. This perspective is well established, pattern recognition is a core of any IQ test. But there is no general AND constructive definition of pattern or recognition, so I have my own in the Outline section.

For excellent popular introductions to cognition-as-prediction thesis see “On Intelligence” by Jeff Hawkins and 
“How to Create a Mind“ by Ray Kurzweil. But on a technical level, they and most everyone else use neural nets, which work in very coarse statistical fashion. Capsule Networks, recently introduced by Geoffrey Hinton et al, are more local and selective by multiple instantiation parameters. But they still start with weighted summation per parameter, which degrades the data before comparison and evaluation.

In the next section, I define similarity for the simplest inputs, then describe hierarchically recursive algorithm 
of search for similarity among incrementally complex inputs: lower-level patterns. The following two sections compare my scheme to ANN, BNN, and CapsNet. This is an open project, we need help with design and implementation: [WIKI](https://github.com/boris-kz/CogAlg/wiki). I pay for contributions or monthly if there is a track record, see [CONTRIBUTING](https://github.com/boris-kz/CogAlg/blob/master/CONTRIBUTING.md). 


### Outline of my approach


Proposed algorithm is a comparison-first alternative to deep learning, neither statistical nor neuromorphic. It is designed to discover patterns of incremental composition in recursively extended pipeline. Each level of this pipeline cross-compares inputs and then clusters them into patterns by proximity and resulting match.

First-level comparands are sensory inputs at the limit of resolution: adjacent pixels of video or equivalents in other modalities. All symbolic data is encoded by some prior cognitive process. To discover meaningful patterns in symbols, they must be decoded before being cross compared. The difficulty of decoding is exponential with the level of encoding, thus starting with raw sensory input is by far the easiest to implement (part 0).

Basic comparison is inverse arithmetic operation between single-variable comparands, of incremental power: Boolean, subtraction, division, etc. Each order of comparison forms miss or loss: XOR, difference, ratio, etc., and match or similarity, which can be defined directly or as inverse deviation of miss. Direct match is compression of represented magnitude by replacing larger input with corresponding order of miss between the inputs: Boolean AND, min input in comp by subtraction, integer part of ratio in comp by division, etc. (part 1). 

These direct similarity measures work if input intensity represents some stable physical property, which anti-correlates with variation in representation. This is the case in tactile but not in visual input: brightness doesn’t correlate with inertia or invariance, dark objects are just as stable as bright ones. So, initial match in vision should be defined indirectly, as inverse deviation of variation in intensity. 1D variation is simply difference, ratio, etc., while multi-D comparison has to combine differences into Euclidean distance and gradient.

Comparison forms patterns, defined here as graphs of matching inputs. In other words, patterns are proximity clusters of input patterns, with above-average nearest-neighbor match, within maximal comparison distance. These filters: average match and maximal distance, are higher-level feedback. Value of representation and prediction = precision of what * precision of where. That “where” is pose: coordinates and dimensions, and we need dimension-specific proximity clustering to represent them (part 2). 
    
Patterns are represented by summed parameters derived from individual comparisons: internal match and miss, plus external continuity and distance. These parameters are selectively cross-compared between input patterns, forming match and miss per parameter, summed into match and miss per pattern, which determine clustering into higher patterns. So, number of parameters per pattern may multiply with each level of search. 

Comparison is selectively incremental, depending on induction from prior comparisons. The increments are in distance and derivation among inputs per pattern, then in pattern composition and parameterization per level. There should be a unique set of operations added per increment, hence a singular in “cognitive algorithm”. Resulting hierarchy is a pipeline: terminated patterns are outputted to the next level, and new level is formed for a pattern terminated by current top level. Which continues as long as the system receives novel inputs. 

Please see [whole-system diagram](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/Whole-system%20hierarchy.png).

Working algorithm will be first level operations plus recursive increment in operations per higher level. This increment is needed to handle incremental syntax (derivatives and dimensions) of outputs relative to inputs.
Higher levels will discover longer-range spatio-temporal and then conceptual patterns, of indefinite generality. But this process is too complex and slow for simple test problems, its design should be driven by theory rather than experimentation. Which is probably why such schemes are not actively explored.

Many readers note disconnect between abstractness of this outline and the amount of detail in current code. That’s because we are in space-time continuum: search must follow proximity in each dimension, which requires very specific processing. Another complaint is that I don't use mathematical notation, but it simply doesn't have the flexibility to express deeply conditional, incrementally complex process. 



### Comparison to Artificial and Biological Neural Networks



All unsupervised learning is some form of pattern discovery. The patterns are defined by match vs error between inputs, which is effectively clustering these inputs into patterns. In my scheme both comparison and clustering are primarily lateral: among inputs within a level, while in statistical learning they are vertical: between layers of composition (weighted summation). That makes all statistical learning some form of [centroid clustering](https://en.wikipedia.org/wiki/Cluster_analysis#Centroid-based_clustering), which conceptually includes neural nets.

Basic ANN is a multi-layer perceptron: each node weighs the inputs at synapses, then sums and thresholds them into output. This normalized sum of inputs is their centroid. Output of top layer is compared to some template, forming an error. With Stochastic Gradient Descent, that error backpropagates, converting initially random weights into functional values. This is a form of learning, but I have basic problems with the process:

- Vertical learning (via feedback of error) takes tens of thousands of cycles to form accurate representations. That's because summation per layer degrades positional input resolution. Hence, the output that drives learning contains exponentially smaller fraction of original information with each added layer. My cross-comp and clustering is far more complex per level, but the output contains all information of the input. Lossy selection is only done on the next level, after evaluation per pattern (vs. before evaluation in statistical methods). 

- Both initial weights and sampling that feeds SGD are randomized. Also driven by random variation are RBMs, GANs, VAEs. But any randomization is antithetical to intelligence, it's only useful in statistical methods, which merge inputs with weights irreversibly. They can’t be separated in the output, so non-random initialization and variation will introduce bias. All input modification in my scheme is via hyperparameters, stored separately and then used to normalize comparisons between inputs formed with different-value hyperparameters. 

- SGD minimizes error (top-layer miss), which is quantitatively different from maximizing match: compression. And that error is wrt. some specific template, while my match is summed over all past input / experience. The “error” here is plural: lateral misses (differences, ratios, etc.), computed by cross-comparison within a level. All inputs represent environment and have positive value. But then they are packed (compressed) into patterns, which have different range and precision, thus different relative value per relatively fixed record cost.

- Representation in ANN is fully distributed, similar to the brain. But the brain has no alternative: there is no substrate for local memory or program in neurons. Computers have RAM, so parallelization is a simple speed vs. efficiency trade-off, useful only for complex semantically isolated nodes. Such nodes are patterns, encapsulating a set of co-derived “what” and “where” parameters. This is similar to neural ensemble, but parameters that are compared together should be localized in memory, not distributed across a network.

More basic neural learning mechanism is Hebbian, though it is rarely used in ML. Conventional spiking version is that weight is increased if the synapse often receives a spike just before the node fires, else the weight is decreased. But input and output don't have to be binary, the same logic can be applied to scalar values: the weight is increased / decreased in proportion to some measure of similarity between its input and following output of the node. That output is normalized sum of all inputs, or their centroid.

Such learning is local, within each node. But it is still a product of vertical comparison: centroid is a higher order of composition than individual inputs. This comparison across composition drives all statistical learning, but it destroys positional information at each layer. Compared to autoencoders: principal backprop-driven unsupervised learning technique, Hebbian learning lacks the decoding stage (as does proposed algorithm). Decoding decomposes hidden layers, to equalize composition orders of output and compared template.

Inspiration by the brain kept ANN research going for decades before they became useful. Their “neurons” are mere stick figures, but that’s not a problem, most of neuron’s complexity is due to constraints of biology. The problem is that core mechanism in ANN: weighted summation, may also be a no-longer needed compensation for such constraints: neural memory requires dedicated connections. That makes representation and cross-comparison of individual inputs nearly impossible, so they are summed. But we now have dirt-cheap RAM.

Other biological constraints are very slow neurons, and the imperative of fast reaction for survival in the wild. Both favor fast though crude summation, at the cost of glacial training. Reaction speed became less important: modern society is quite secure, while continuous learning is far more important because of accelerating progress. Summation also reduces noise, which is very important for neurons that often fire at random, to initiate and maintain latent connections. But that’s irrelevant for electronic circuits.

Evolution is extremely limited in the amount of complexity that can be added before it is pruned by natural selection. I see no way it could produce something as complex as proposed algorithm. Biological intelligence is a distant side effect of maximizing reproduction. The brain evolved to guide the body, with neurons originating as instinctive stimulus-to-response converters. They only do pattern discovery as an instrumental upshot, not a primary purpose. Hence, both SGD and Hebbian learning is fitting, driven by feedback of action, triggered by weighted input sum. 

Uri Hasson, Samuel Nastase, Ariel Goldstein reach a similar conclusion in “Direct fit to nature: an evolutionary perspective on biological and artificial neural networks”: “We argue that neural computation is grounded in brute-force direct fitting, which relies on over-parameterized optimization algorithms to increase predictive power (generalization) without explicitly modeling the underlying generative structure of the world. Although ANNs are indeed highly simplified models of BNNs, they belong to the same family of over-parameterized, direct-fit models, producing solutions that are mistakenly interpreted in terms of elegant design principles but in fact reflect the interdigitation of ‘‘mindless’’ optimization processes and the structure of the world.”


### Comparison to Capsule Networks


The nearest experimentally successful method is recently introduced “capsules”. Some similarities to CogAlg:
- capsules also output multivariate vectors, “encapsulating” several parameters, similar to my patterns,
- these parameters also include pose: coordinates and dimensions, compared to compute corresponding miss,
- these misses / distances are compared to find affine transformations or equivariance: my match of misses,
- capsules also send direct feedback to lower layer: dynamic routing, vs. trans-hidden-layer backprop in ANN.

My main problems with CapsNet and alternative treatment:
 
- Object is defined as a recurring configuration of different parts. But such recurrence can’t be assumed, it should be derived by cross-comparing relative position among parts of matching objects. Which can only be done after their positions are cross-compared, which is after their objects are cross-compared: two levels above the level that forms initial objects. So, objects formed by positional equivariance would be secondary, though they may displace initial segmentation objects as a primary representation. Stacked Capsule Autoencoders also have exclusive segmentation on the first layer, but proximity doesn’t matter on their higher layers.

- Routing by agreement is basically recursive input clustering, by match of input vector to the output vector. The output (centroid) represents inputs at all locations, so its comparison to inputs is effectively mixed-distance. Thus, clustering in CapsNet is fuzzy and discontinuous, forming redundant representations. Routing by agreement reduces that redundancy, but not consistently so, it doesn’t specifically account for it. 
My default clustering is exclusive segmentation: each element (child) belongs to only one cluster (parent). Fuzzy clustering is selective to inputs valued above the cost of adjusting for overlap in representation, which increases with the range of cross-comparison. Conditional range increase is done on all levels of composition

- Instantiation parameters are application-specific, CapsNet has no general mechanism to derive them. My general mechanism is cross-comparison of input capsule parameters, which forms higher-order parameters. First level forms pixel-level gradient, similar to edge detection in CNN. But then it forms proximity-constrained clusters, defined by gradient and parameterized by summed pixel intensity, dy, dx, gradient, angle. This cross-comparison followed by clustering is done on all levels, with incremental number of parameters per input.

- Number of layers is fixed, while I think it should be incremental with experience. My hierarchy is a dynamic pipeline: patterns are displaced from a level by criterion sign change and sent to existing or new higher level. So, both hierarchy of patterns per system and sub-hierarchy of derivatives per pattern expand with experience. The derivatives are summed within a pattern, then evaluated for extending intra-pattern search and feedback.

- Output vector of higher capsules combines parameters of all lower layers into Euclidean distance. That is my default too, but they should also be kept separate, for potential cross-comp among layer-wide representations.
Overall, CapsNet is a variation of ANN, with input summation first and dynamic routing second. So, it’s a type of Hebbian learning, with most of the problems that I listed in the previous section.



### Quantifying match and miss between variables



The purpose here is prediction, and predictive value is usually defined as [compressibility](https://en.wikipedia.org/wiki/Algorithmic_information_theory). Which is perfectly fine, but existing methods only compute compression per sequence of inputs. To enable more incremental selection and scalable search, I quantify partial match between atomic inputs, vs. binary same | different choice for inputs within sequences. This is similar to the way Bayesian inference improved on classical logic, by quantifying probability vs. binary true | false values.

Partial match between two variables is a complementary of miss, in corresponding power of comparison: 
- Boolean match is AND and miss is XOR (two zero inputs form zero match and zero miss), 
- comparison by subtraction increases match to a smaller comparand and reduces miss to a difference,
- comparison by division increases match to min * integer part of ratio and reduces miss to a fractional part
(direct match works for tactile input. but reflected-light in vision requires inverse definition of initial match)

In other words, match is a compression of larger comparand’s magnitude by replacing it with miss. Which means that match = smaller input: a common subset of both inputs, = sum of AND between their uncompressed (unary code) representations. Ultimate criterion is recorded magnitude, rather than bits of memory it occupies, because the former represents physical impact that we want to predict. The volume of memory used to record that magnitude depends on prior compression, which is not an objective parameter. 

Given incremental complexity, initial inputs should have binary resolution and implicit shared coordinate (being a macro-parameter, resolution of coordinate lags that of an input). Compression of bit inputs by AND is well known as digitization: substitution of two lower 1 bits with one higher 1 bit. Resolution of coordinate (input summation span) is adjusted by feedback to form integers that are large enough to produce above-average match.

Next-order compression can be achieved by comparison between consecutive integers, distinguished by binary (before | after) coordinate. Basic comparison is inverse arithmetic operation of incremental power: AND, subtraction, division, logarithm, and so on. Additive match is achieved by comparison of a higher power than that which produced comparands: comparison by AND will not further compress integers previously digitized by AND.

Rather, initial comparison between integers is by subtraction, resulting difference is miss, and absolute match is a smaller input. For example, if inputs are 4 and 7, then miss is 3, and their match or common subset is 4. Difference is smaller than XOR (non-zero complementary of AND) because XOR may include opposite-sign (opposite-direction) bit pairs 0, 1 and 1, 0, which are cancelled-out by subtraction.

Comparison by division forms ratio, which is a magnitude-compressed difference. This compression is explicit in long division: match is accumulated over iterative subtraction of smaller comparand from remaining difference. In other words, this is also a comparison by subtraction, but between different orders of derivation. Resulting match is smaller comparand * integer part of ratio, and miss is final reminder or fractional part of ratio.

Ratio can be further compressed by converting to radix | logarithm, and so on. But computational costs may grow even faster. Thus, power of comparison should increase only for inputs sufficiently compressed by lower power: AND for bit inputs, SUB for integer inputs, DIV for pattern inputs, etc. Actual compression depends on input and on resolution of its coordinate: input | derivative summation span. We can’t control the input, so average match is adjusted via resolution of coordinate.

To filter future inputs, this absolute match should be projected: recombined with co-derived miss projected for a target distance. Filter deviation is accumulated until it exceeds the cost of updating lower-level filter. Which then forms relative match: current match - past match that co-occurs with average higher-level projected match. This relative match: above- or below- average predictive value, determines input inclusion into positive or negative predictive value pattern.

Separate filters are formed for each type of compared variable. Initial input, such as reflected light, is likely to be incidental and very indirectly representative of physical properties in observed objects. Then its filter will increase, reducing number of positive patterns, potentially down to 0. But differences or ratios between inputs represent variation, which is anti-correlated with match. They have negative predictive value, inverted to get incrementally closer to intrinsically predictive properties, such as mass or momentum.

Hence a vision-specific way I define initial match. Predictive visual property is albedo, which means locally stable ratio of brightness / intensity. Since lighting is usually uniform over much larger area than pixel, the difference in brightness between adjacent pixels should also be stable. Relative brightness indicates some underlying property, so it should be cross-compared to form patterns. But it is reflected: only indirectly representative of observed object.

Absent significant correlation between input magnitude and represented physical object magnitude, the only proxy to match in initial comparison is inverse deviation of absolute difference:
average_|difference| - |difference|. Though less accurate (defined via average diff vs. individual input), this match is also a complementary of diff:
- complementary of |difference| within average_|difference| (=max of the |difference| s), similar to minimum:
- complementary of |difference| within max input.



### Implementation



Any prediction has two components: what and where. We must have both: value of prediction = precision of what * precision of where. That “where” is currently neglected: statistical ML methods represent coordinates much more coarsely than the inputs. Hence, precision of where (spans of and distances between patterns) is degraded, and so is predictive value of combined representations. That's not the case here because my top-level patterns (multi-dimensional blobs) are contiguous.

Core algorithm is 1D: time only. Our space-time is 4D, and average match is presumably equal over all dimensions. That means patterns defined in fewer dimensions will be only slices of actual input, fundamentally limited and biased by the angle of scanning / slicing. Hence, initial pixel comparison should also be over 4D at once, or at least over 3D for video and 2D for still images. This full-D-cycle level of search is a universe-specific extension of core algorithm. The dimensions should be discoverable by the core algorithm, but coding it in is much faster. 

This repository currently has three versions of 1st D-cycle, analogous to connected-component analysis: 1D line alg, 2D frame alg, and 3D video alg.
Subsequent cycles will compare full-D-terminated input patterns over increasing distance in each dimension, forming discontinuous patterns of incremental composition and range.
“Dimension” here defines external sequence and distance among inputs. This is different from conventional clustering, which treats both external and internal parameters as dimensions. 

Complete hierarchical algorithm will have two-level code: 
- 1st level algorithm: contiguous cross-comparison over full-D cycle, plus feedback to adjust most and least significant bits of the input. 
- Recurrent increment in complexity, extending current-level alg to next-level alg. This increment will account for increasing internal complexity of input patterns on higher levels, unfolding them for cross-comparison and re-folding results for evaluation and feedback.

Initial testing could be on recognition of labeled images, but video or stereo video should be much better. We will then add colors, maybe audio and text. 

For more detailed account of current development see [WIKI](https://github.com/boris-kz/CogAlg/wiki).

Suggestions and collaboration are most welcome, see [CONTRIBUTING](https://github.com/boris-kz/CogAlg/blob/master/CONTRIBUTING.md).


