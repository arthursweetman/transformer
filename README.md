# Transformer

This is a LLM built using the transformer architecture from scratch using Pytorch and follows the transformer architecture
that was described in the 2017 paper titled *Attention is All you Need*.

I built a decoder-only model (like is used in ChatGPT) and trained it on the works of Shakespeare (see *input.txt*).
After training this on my local cpu over one night, I was able to acheive some modest results.
The model can output infinite Shakespeare-esque text, although at the moment it does not
actually make any sense. The point here is to show that the architecture I have implemented has been 
successful and the only bottlenecks now are **scaling up the model** and **increasing computing power**.

However, to see that I am on the right track, this is an example of what the model can output
from a random request:

```
ESCALUS:
fie, this and the kings of the since,
And with his Mowwbrathe made in have healned; the deem you, the crow cals deces upose.

KING LEWIS XI:
Well, that is men.

CORIOLANUS:
And, fair part, Sarciater.
Beforth and gains the Caiusamin, score, and sweet my red, fortunest counce for your brish! and year.

APULET:
Prevended thy cushor,
Or staings is cousin his break of montly show belike us I am signak, as he these have do did spent,
Nost ceever you and hence, by paned her done. and.
You bot
```

There is clear room for improvement here which could be acheived by breaking through the 
bottlenecks I listed above.

I'd like to give a special thanks to [Andrej Karpathy](https://karpathy.ai/) for his contribution 
to the teaching of LLMs and for providing the [guidance](https://youtu.be/kCc8FmEb1nY?si=ilIH-duM_vJavVKv) for me on this project.