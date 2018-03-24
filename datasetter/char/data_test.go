package char

const corpus = `
The recent development of various methods of modulation such as PCM and PPM which exchange
bandwidth for signal-to-noise ratio has intensified the interest in a general theory of communication. A
basis for such a theory is contained in the important papers of Nyquist and Hartley on this subject. In the
present paper we will extend the theory to include a number of new factors, in particular the effect of noise
in the channel, and the savings possible due to the statistical structure of the original message and due to the
nature of the final destination of the information.
The fundamental problem of communication is that of reproducing at one point either exactly or approximately
a message selected at another point. Frequently the messages have meaning; that is they refer
to or are correlated according to some system with certain physical or conceptual entities. These semantic
aspects of communication are irrelevant to the engineering problem. The significant aspect is that the actual
message is one selected from a set of possible messages. The system must be designed to operate for each
possible selection, not just the one which will actually be chosen since this is unknown at the time of design.
If the number of messages in the set is finite then this number or any monotonic function of this number
can be regarded as a measure of the information produced when one message is chosen from the set, all
choices being equally likely. As was pointed out by Hartley the most natural choice is the logarithmic
function. Although this definition must be generalized considerably when we consider the influence of the
statistics of the message and when we have a continuous range of messages, we will in all cases use an
essentially logarithmic measure.
`
