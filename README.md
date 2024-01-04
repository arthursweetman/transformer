# Transformer

This is a LLM built using the transformer architecture from scratch using Pytorch and follows the transformer architecture
that was described in the 2017 paper titled *Attention is All you Need*.

I built a decoder-only model (like is used in ChatGPT) and trained it on the works of Shakespeare (see *input.txt*).
After training this on a GPU for 10 minutes, I was able to acheive some rather good results.
The model can output infinite Shakespeare-esque text. Full words are outputted and the sentence structure does almost seem like it was written by Shakespeare himself. 

The point here is to show that the architecture I have implemented has been successful. And my intention is to expand on this project in the future to perform bigger and better tasks.

 Feel free to visit [my website](https://www.arthur-sweetman.com/artificial-intelligence) to generate infinite Shakespeare using this trained LLM.

```
Are my light-trial in thy lip,
Since without thy love's brain?
Yet, nothing; let me frail
I do do the new-or stones
By thy power, Margaret!

GLOUCESTER:
The plate that come of it,
And the some sour manner! My neighbours: how oft?

BISHOP OF ELY:
And we my royal friends,
Whose rote the king?

GREY:
Ay, but see, fetch other,
Before thee fourteen of: the 'twas sentence
With dubband sentious chip of devils, may saint
Amongst my true my hate;
But only they plead and joyn of thine eyes and throat--in
```

I'd like to give a special thanks to [Andrej Karpathy](https://karpathy.ai/) for his contribution 
to the teaching of LLMs and for providing the [guidance](https://youtu.be/kCc8FmEb1nY?si=ilIH-duM_vJavVKv) for me on this project.