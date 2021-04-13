

## experimenting with LDA, on a small dataset.
- 5th of Feb, 2021, Just started reading GreenLights. A great read.
- used lda
- (idiom, context) columns
- num_topics = 20
- iterations = 50
- passes =  3

```
INFO:gensim.models.ldamodel:topic diff=0.033702, rho=0.120718
DEBUG:gensim.models.ldamodel:bound: at document #0
INFO:gensim.models.ldamodel:-10.435 per-word bound, 1384.5 perplexity estimate based on a held-out corpus of 1242 documents with 4695 words
0 ---> 0.201*"want" + 0.050*"talk" + 0.031*"long" + 0.030*"fuck" + 0.023*"care" + 0.022*"family" + 0.016*"change" + 0.014*"eat" + 0.011*"think" + 0.010*"tell"
1 ---> 0.136*"look" + 0.043*"ask" + 0.038*"like" + 0.026*"dad" + 0.024*"sit" + 0.022*"lie" + 0.017*"room" + 0.016*"forget" + 0.016*"plan" + 0.014*"ai"
2 ---> 0.219*"right" + 0.124*"hey" + 0.020*"help" + 0.018*"pay" + 0.017*"easy" + 0.015*"think" + 0.013*"shoot" + 0.013*"what was that" + 0.012*"clear" + 0.012*"cool"
3 ---> 0.107*"time" + 0.068*"work" + 0.032*"happen" + 0.023*"get" + 0.020*"break" + 0.018*"face" + 0.014*"tell" + 0.013*"move" + 0.012*"call" + 0.010*"mrs"
4 ---> 0.140*"na" + 0.140*"gon" + 0.095*"need" + 0.035*"listen" + 0.030*"tell" + 0.015*"matter" + 0.014*"check" + 0.012*"think" + 0.012*"set" + 0.011*"tomorrow"
5 ---> 0.178*"oh" + 0.063*"uh" + 0.061*"leave" + 0.052*"wait" + 0.033*"god" + 0.020*"huh" + 0.020*"hi" + 0.015*"ah" + 0.013*"guess" + 0.011*"think"
6 ---> 0.070*"say" + 0.051*"love" + 0.047*"go" + 0.029*"old" + 0.025*"believe" + 0.020*"die" + 0.020*"lose" + 0.015*"think" + 0.014*"ok" + 0.014*"crazy"
7 ---> 0.067*"thing" + 0.034*"year" + 0.031*"guy" + 0.026*"make" + 0.026*"understand" + 0.023*"you know what" + 0.022*"see" + 0.022*"remember" + 0.021*"hold" + 0.019*"think"
8 ---> 0.050*"hear" + 0.042*"kill" + 0.030*"got" + 0.027*"take" + 0.023*"mr" + 0.023*"ta" + 0.022*"course" + 0.022*"problem" + 0.021*"head" + 0.015*"fucking"
9 ---> 0.071*"♪" + 0.052*"stop" + 0.040*"kid" + 0.032*"hell" + 0.023*"everybody" + 0.021*"ready" + 0.019*"fall" + 0.016*"whoa" + 0.016*"drink" + 0.013*"sign"
10 ---> 0.154*"come" + 0.063*"little" + 0.044*"feel" + 0.028*"like" + 0.025*"well" + 0.025*"wo" + 0.024*"fine" + 0.017*"watch" + 0.016*"shit" + 0.015*"bit"
11 ---> 0.045*"find" + 0.033*"night" + 0.031*"run" + 0.028*"nice" + 0.027*"sir" + 0.026*"car" + 0.024*"walk" + 0.022*"stand" + 0.017*"man" + 0.017*"hello"
12 ---> 0.091*"yes" + 0.032*"idea" + 0.019*"tonight" + 0.018*"open" + 0.016*"door" + 0.016*"lot" + 0.014*"think" + 0.014*"phone" + 0.014*"true" + 0.013*"fun"
13 ---> 0.084*"thank" + 0.027*"mother" + 0.026*"miss" + 0.024*"mom" + 0.018*"happy" + 0.017*"worry" + 0.016*"school" + 0.014*"what's going on" + 0.014*"book" + 0.013*"money"
14 ---> 0.257*"know" + 0.100*"let" + 0.098*"okay" + 0.050*"think" + 0.022*"maybe" + 0.021*"way" + 0.019*"home" + 0.017*"wrong" + 0.011*"sleep" + 0.010*"people"
15 ---> 0.142*"like" + 0.078*"mean" + 0.026*"hand" + 0.019*"baby" + 0.017*"speak" + 0.016*"sound" + 0.011*"hope" + 0.010*"think" + 0.009*"couple" + 0.008*"wow"
16 ---> 0.047*"great" + 0.017*"excuse" + 0.016*"wish" + 0.013*"promise" + 0.011*"hide" + 0.010*"blood" + 0.010*"perfect" + 0.010*"control" + 0.010*"funny" + 0.009*"use"
17 ---> 0.188*"get" + 0.017*"son" + 0.011*"pull" + 0.011*"gun" + 0.011*"fight" + 0.010*"try" + 0.010*"go" + 0.010*"party" + 0.009*"somebody" + 0.009*"find"
18 ---> 0.123*"good" + 0.034*"friend" + 0.028*"bad" + 0.025*"life" + 0.020*"real" + 0.018*"exactly" + 0.017*"minute" + 0.016*"father" + 0.016*"morning" + 0.014*"hard"
19 ---> 0.180*"yeah" + 0.060*"sure" + 0.057*"sorry" + 0.023*"mind" + 0.021*"boy" + 0.019*"child" + 0.018*"hour" + 0.012*"day" + 0.010*"spend" + 0.008*"think"
```


cannot really discern one topic from another...
is this when I need to look into topic coherence?

## next experiment

just..let's try with 100 topics. with more passes. then, we may be able to find some insights.
- the same idiom, context dataset (context_length=1. i.e. Just the sentence itself)
```python
    # you must pass the.. encoded version of bows. (they must be integers)
    lda_model = LdaMulticore(bows,
                             workers=4,
                             # for debugging and topic printing, we need to give it this.
                             id2word=dct,
                             num_topics=100,
                             passes=10)
    # have a guess, what are those topics? - how do I see them...?
    for topic_id, idx2prob in lda_model.show_topics(num_topics=100, num_words=10, formatted=True):
        print(topic_id, "--->", idx2prob)

```

```
INFO:gensim.models.ldamodel:topic diff=0.018812, rho=0.114995
DEBUG:gensim.models.ldamodel:bound: at document #0
INFO:gensim.models.ldamodel:-27.044 per-word bound, 138421510.1 perplexity estimate based on a held-out corpus of 1242 documents with 4695 words
0 ---> 0.384*"way" + 0.137*"guess" + 0.069*"drink" + 0.041*"know" + 0.037*"worth" + 0.030*"think" + 0.025*"name" + 0.024*"owe" + 0.017*"chase" + 0.016*"dumb"
1 ---> 0.227*"night" + 0.168*"understand" + 0.091*"ready" + 0.090*"hello" + 0.037*"information" + 0.034*"future" + 0.023*"large" + 0.022*"convince" + 0.019*"address" + 0.017*"on the ground"
2 ---> 0.352*"uh" + 0.124*"problem" + 0.038*"jesus" + 0.036*"luck" + 0.031*"fault" + 0.028*"bill" + 0.024*"pretend" + 0.022*"al" + 0.019*"ben" + 0.017*"opinion"
3 ---> 0.163*"matter" + 0.079*"prove" + 0.070*"burn" + 0.049*"busy" + 0.048*"raise" + 0.038*"taste" + 0.033*"conversation" + 0.033*"opportunity" + 0.025*"demand" + 0.019*"crowd"
4 ---> 0.210*"fine" + 0.055*"parent" + 0.055*"bye" + 0.035*"store" + 0.030*"blame" + 0.023*"out of it" + 0.022*"church" + 0.022*"afternoon" + 0.019*"ma" + 0.017*"butt"
5 ---> 0.157*"place" + 0.121*"play" + 0.089*"send" + 0.088*"end" + 0.074*"easy" + 0.054*"question" + 0.037*"record" + 0.033*"weapon" + 0.030*"client" + 0.027*"short"
6 ---> 0.400*"yeah" + 0.346*"tell" + 0.037*"use" + 0.036*"word" + 0.023*"truth" + 0.014*"joke" + 0.014*"think" + 0.013*"daddy" + 0.013*"calm" + 0.010*"waste"
7 ---> 0.387*"let" + 0.152*"sure" + 0.108*"stop" + 0.085*"wo" + 0.057*"walk" + 0.037*"door" + 0.022*"safe" + 0.022*"know" + 0.018*"bag" + 0.013*"building"
8 ---> 0.135*"baby" + 0.129*"dead" + 0.076*"police" + 0.060*"welcome" + 0.054*"explain" + 0.052*"early" + 0.049*"past" + 0.033*"tired" + 0.033*"sam" + 0.031*"appear"
9 ---> 0.388*"time" + 0.114*"new" + 0.093*"make" + 0.041*"spend" + 0.026*"think" + 0.019*"george" + 0.019*"quick" + 0.019*"grab" + 0.019*"court" + 0.018*"know"
10 ---> 0.060*"kick" + 0.052*"suck" + 0.051*"join" + 0.046*"continue" + 0.040*"green" + 0.039*"tv" + 0.035*"tear" + 0.026*"justice" + 0.025*"french" + 0.024*"clark"
11 ---> 0.235*"turn" + 0.122*"afraid" + 0.057*"notice" + 0.050*"charlie" + 0.045*"boyfriend" + 0.039*"disappear" + 0.038*"getting" + 0.035*"lovely" + 0.029*"bone" + 0.020*"tight"
12 ---> 0.510*"okay" + 0.115*"old" + 0.105*"got" + 0.097*"ta" + 0.029*"possible" + 0.014*"know" + 0.009*"-you" + 0.009*"obvious" + 0.008*"think" + 0.007*"advantage"
13 ---> 0.333*"little" + 0.211*"see" + 0.061*"chance" + 0.056*"lady" + 0.037*"think" + 0.031*"bar" + 0.028*"know" + 0.018*"wine" + 0.015*"hire" + 0.014*"army"
14 ---> 0.097*"what was that" + 0.085*"mrs" + 0.075*"touch" + 0.055*"relationship" + 0.053*"hospital" + 0.052*"key" + 0.050*"group" + 0.034*"field" + 0.033*"patient" + 0.026*"piss"
15 ---> 0.168*"hi" + 0.113*"gun" + 0.081*"city" + 0.065*"handle" + 0.064*"hair" + 0.046*"bottle" + 0.040*"file" + 0.031*"favorite" + 0.027*"plenty" + 0.024*"unfortunately"
16 ---> 0.244*"sorry" + 0.179*"believe" + 0.075*"happy" + 0.055*"pull" + 0.049*"honey" + 0.042*"cop" + 0.039*"control" + 0.033*"know" + 0.029*"ring" + 0.025*"list"
17 ---> 0.101*"dude" + 0.073*"mention" + 0.068*"uncle" + 0.056*"as soon as" + 0.051*"escape" + 0.041*"shock" + 0.041*"event" + 0.040*"race" + 0.029*"approach" + 0.024*"i'ii"
18 ---> 0.206*"home" + 0.201*"fuck" + 0.128*"head" + 0.055*"hide" + 0.027*"difficult" + 0.027*"know" + 0.025*"relax" + 0.020*"cook" + 0.019*"beer" + 0.016*"discover"
19 ---> 0.145*"tonight" + 0.136*"sleep" + 0.117*"body" + 0.056*"smart" + 0.055*"willing" + 0.050*"fix" + 0.039*"major" + 0.029*"forgive" + 0.019*"occur" + 0.016*"represent"
20 ---> 0.622*"get" + 0.141*"day" + 0.028*"couple" + 0.024*"know" + 0.020*"bitch" + 0.017*"cause" + 0.015*"partner" + 0.012*"pop" + 0.011*"mama" + 0.009*"pressure"
21 ---> 0.173*"meet" + 0.096*"check" + 0.093*"write" + 0.085*"order" + 0.077*"rest" + 0.068*"answer" + 0.031*"spot" + 0.029*"bank" + 0.019*"gang" + 0.018*"military"
22 ---> 0.058*"prison" + 0.053*"christ" + 0.040*"some kind of" + 0.039*"access" + 0.039*"prepare" + 0.033*"jeff" + 0.032*"energy" + 0.032*"princess" + 0.031*"nearly" + 0.030*"jerk"
23 ---> 0.148*"watch" + 0.105*"fall" + 0.105*"fucking" + 0.051*"force" + 0.035*"obviously" + 0.034*"space" + 0.033*"serve" + 0.027*"smile" + 0.023*"scene" + 0.023*"paul"
24 ---> 0.257*"take" + 0.109*"exactly" + 0.068*"learn" + 0.056*"win" + 0.053*"show" + 0.053*"far" + 0.035*"ship" + 0.031*"arrest" + 0.026*"board" + 0.017*"think"
25 ---> 0.134*"everybody" + 0.106*"able" + 0.091*"inside" + 0.069*"sense" + 0.055*"consider" + 0.049*"buddy" + 0.044*"totally" + 0.036*"letter" + 0.028*"double" + 0.027*"mountain"
26 ---> 0.183*"pay" + 0.052*"terrible" + 0.050*"crime" + 0.041*"low" + 0.038*"dick" + 0.035*"evil" + 0.026*"shop" + 0.025*"commit" + 0.023*"roger" + 0.023*"awful"
27 ---> 0.214*"well" + 0.173*"lot" + 0.125*"nice" + 0.098*"probably" + 0.096*"huh" + 0.031*"think" + 0.018*"jail" + 0.017*"thing" + 0.016*"billy" + 0.013*"ridiculous"
28 ---> 0.135*"dad" + 0.132*"house" + 0.110*"mom" + 0.061*"dr" + 0.055*"husband" + 0.043*"secret" + 0.037*"report" + 0.035*"alive" + 0.034*"teach" + 0.032*"completely"
29 ---> 0.242*"um" + 0.100*"stupid" + 0.095*"'em" + 0.036*"dangerous" + 0.028*"fan" + 0.022*"village" + 0.020*"think" + 0.018*"nonsense" + 0.017*"struggle" + 0.015*"guide"
30 ---> 0.423*"come" + 0.320*"look" + 0.036*"plan" + 0.030*"know" + 0.026*"whoa" + 0.019*"funny" + 0.019*"steal" + 0.014*"think" + 0.008*"thing" + 0.008*"plane"
31 ---> 0.337*"start" + 0.072*"outside" + 0.038*"tell you what" + 0.033*"kevin" + 0.020*"matt" + 0.020*"sunday" + 0.018*"plate" + 0.017*"unit" + 0.015*"think" + 0.014*"tuesday"
32 ---> 0.371*"year" + 0.067*"bet" + 0.050*"old" + 0.043*"king" + 0.025*"view" + 0.022*"chuckle" + 0.022*"van" + 0.020*"restaurant" + 0.019*"responsible" + 0.017*"leader"
33 ---> 0.274*"mr" + 0.185*"bring" + 0.040*"american" + 0.028*"roll" + 0.027*"machine" + 0.027*"fish" + 0.026*"rich" + 0.024*"support" + 0.024*"career" + 0.019*"snap"
34 ---> 0.156*"car" + 0.090*"business" + 0.080*"ass" + 0.076*"what's going on" + 0.070*"hang" + 0.046*"officer" + 0.046*"begin" + 0.034*"meeting" + 0.034*"know" + 0.031*"treat"
35 ---> 0.326*"work" + 0.198*"stay" + 0.075*"eat" + 0.045*"dog" + 0.042*"bear" + 0.023*"mmm" + 0.019*"press" + 0.017*"huge" + 0.017*"concern" + 0.016*"know"
36 ---> 0.295*"guy" + 0.256*"try" + 0.077*"today" + 0.040*"cut" + 0.034*"country" + 0.033*"know" + 0.023*"think" + 0.022*"mess" + 0.019*"fear" + 0.013*"smoke"
37 ---> 0.362*"hear" + 0.136*"child" + 0.027*"soul" + 0.025*"spirit" + 0.024*"tommy" + 0.022*"jame" + 0.020*"player" + 0.020*"football" + 0.016*"emergency" + 0.015*"julia"
38 ---> 0.062*"lay" + 0.048*"area" + 0.041*"cat" + 0.041*"rise" + 0.039*"simply" + 0.031*"cheat" + 0.031*"slip" + 0.029*"desk" + 0.029*"richard" + 0.028*"health"
39 ---> 0.288*"wait" + 0.187*"big" + 0.127*"'cause" + 0.122*"course" + 0.018*"pleasure" + 0.015*"johnny" + 0.013*"fancy" + 0.013*"nah" + 0.011*"bell" + 0.011*"miracle"
40 ---> 0.301*"die" + 0.074*"mike" + 0.054*"ray" + 0.043*"floor" + 0.042*"band" + 0.041*"kinda" + 0.030*"artist" + 0.025*"normally" + 0.021*"toss" + 0.021*"assistant"
41 ---> 0.164*"have" + 0.109*"marry" + 0.085*"war" + 0.084*"bed" + 0.049*"dress" + 0.028*"dave" + 0.025*"don" + 0.023*"think" + 0.023*"eventually" + 0.022*"subject"
42 ---> 0.075*"~" + 0.053*"history" + 0.044*"jimmy" + 0.042*"upstairs" + 0.040*"robert" + 0.037*"guest" + 0.035*"hardly" + 0.025*"london" + 0.025*"warrant" + 0.023*"technology"
43 ---> 0.130*"hour" + 0.124*"face" + 0.080*"later" + 0.054*"street" + 0.037*"accept" + 0.028*"gold" + 0.027*"gift" + 0.026*"normal" + 0.026*"yo" + 0.022*"egg"
44 ---> 0.399*"na" + 0.397*"gon" + 0.016*"experience" + 0.015*"ice" + 0.014*"know" + 0.013*"suit" + 0.010*"magic" + 0.009*"wood" + 0.009*"weekend" + 0.008*"sergeant"
45 ---> 0.610*"like" + 0.130*"feel" + 0.045*"know" + 0.025*"beautiful" + 0.022*"beat" + 0.021*"think" + 0.016*"music" + 0.013*"count" + 0.011*"killer" + 0.011*"photo"
46 ---> 0.263*"happen" + 0.111*"wrong" + 0.102*"mind" + 0.084*"deal" + 0.055*"half" + 0.044*"grow" + 0.036*"foot" + 0.030*"know" + 0.027*"box" + 0.023*"idiot"
47 ---> 0.384*"need" + 0.132*"girl" + 0.089*"family" + 0.045*"hurt" + 0.042*"jack" + 0.027*"sick" + 0.025*"know" + 0.018*"judge" + 0.018*"general" + 0.014*"hotel"
48 ---> 0.284*"boy" + 0.144*"ok" + 0.076*"date" + 0.056*"age" + 0.037*"interesting" + 0.020*"english" + 0.019*"immediately" + 0.015*"financial" + 0.015*"interrupt" + 0.015*"creep"
49 ---> 0.394*"talk" + 0.068*"heart" + 0.068*"doctor" + 0.024*"scream" + 0.021*"know" + 0.021*"take it easy" + 0.020*"goddamn" + 0.020*"buck" + 0.019*"chuck" + 0.018*"driver"
50 ---> 0.673*"right" + 0.024*"choose" + 0.019*"know" + 0.017*"usually" + 0.016*"tough" + 0.015*"message" + 0.015*"thing" + 0.015*"leg" + 0.012*"honestly" + 0.012*"t"
51 ---> 0.175*"story" + 0.071*"jump" + 0.056*"cold" + 0.039*"forward" + 0.035*"possibly" + 0.030*"cousin" + 0.030*"ought" + 0.026*"don't tell me" + 0.022*"motherfucker" + 0.021*"connection"
52 ---> 0.130*"school" + 0.095*"high" + 0.091*"blood" + 0.084*"party" + 0.069*"swear" + 0.047*"dance" + 0.041*"knock" + 0.033*"sing" + 0.022*"computer" + 0.021*"shake"
53 ---> 0.128*"week" + 0.106*"open" + 0.088*"move" + 0.086*"hate" + 0.077*"tomorrow" + 0.075*"late" + 0.070*"wear" + 0.045*"dinner" + 0.028*"window" + 0.027*"mission"
54 ---> 0.101*"shut" + 0.096*"anymore" + 0.055*"wonderful" + 0.041*"dig" + 0.038*"chicken" + 0.033*"impossible" + 0.030*"lieutenant" + 0.029*"solve" + 0.024*"confess" + 0.021*"beginning"
55 ---> 0.537*"go" + 0.092*"hope" + 0.046*"wish" + 0.042*"shoot" + 0.029*"protect" + 0.025*"offer" + 0.025*"cover" + 0.019*"think" + 0.018*"respect" + 0.017*"gentleman"
56 ---> 0.119*"crazy" + 0.073*"pass" + 0.049*"system" + 0.045*"paper" + 0.043*"amazing" + 0.038*"rock" + 0.034*"blue" + 0.027*"chair" + 0.025*"cheer" + 0.025*"jane"
57 ---> 0.122*"drive" + 0.100*"save" + 0.096*"important" + 0.088*"water" + 0.070*"white" + 0.041*"fill" + 0.040*"dark" + 0.033*"note" + 0.027*"difference" + 0.025*"folk"
58 ---> 0.458*"hey" + 0.162*"kill" + 0.090*"lose" + 0.059*"throw" + 0.028*"food" + 0.021*"deep" + 0.021*"club" + 0.012*"know" + 0.010*"barely" + 0.008*"west"
59 ---> 0.336*"people" + 0.160*"listen" + 0.076*"fight" + 0.046*"trouble" + 0.034*"know" + 0.034*"clean" + 0.030*"shot" + 0.029*"brain" + 0.021*"stone" + 0.019*"drunk"
60 ---> 0.375*"thank" + 0.071*"reason" + 0.059*"number" + 0.033*"admit" + 0.032*"earth" + 0.028*"strong" + 0.027*"decision" + 0.020*"interested" + 0.019*"interest" + 0.018*"action"
61 ---> 0.237*"money" + 0.076*"state" + 0.050*"test" + 0.036*"belong" + 0.021*"secretary" + 0.020*"united" + 0.020*"dozen" + 0.020*"community" + 0.019*"devil" + 0.019*"operation"
62 ---> 0.115*"set" + 0.104*"hmm" + 0.081*"mm" + 0.055*"arm" + 0.040*"issue" + 0.039*"wedding" + 0.034*"extra" + 0.033*"basically" + 0.025*"one's" + 0.025*"peace"
63 ---> 0.141*"hold" + 0.139*"world" + 0.132*"son" + 0.086*"pick" + 0.046*"ahead" + 0.040*"charge" + 0.039*"imagine" + 0.038*"class" + 0.036*"card" + 0.031*"detective"
64 ---> 0.179*"ai" + 0.117*"fly" + 0.088*"enjoy" + 0.078*"attack" + 0.057*"forever" + 0.025*"rain" + 0.024*"wash" + 0.023*"realise" + 0.023*"diamond" + 0.022*"purpose"
65 ---> 0.166*"care" + 0.122*"case" + 0.096*"excuse" + 0.060*"office" + 0.056*"wow" + 0.049*"news" + 0.037*"animal" + 0.028*"victim" + 0.021*"cost" + 0.018*"in the middle of"
66 ---> 0.351*"life" + 0.092*"eye" + 0.052*"sex" + 0.049*"one" + 0.043*"worried" + 0.029*"strange" + 0.022*"know" + 0.022*"gay" + 0.021*"remind" + 0.020*"match"
67 ---> 0.064*"red" + 0.053*"frank" + 0.049*"witness" + 0.041*"government" + 0.039*"accident" + 0.036*"beg" + 0.028*"complete" + 0.026*"plus" + 0.022*"congratulation" + 0.022*"flight"
68 ---> 0.411*"want" + 0.157*"know" + 0.062*"kind" + 0.054*"thing" + 0.046*"minute" + 0.043*"real" + 0.039*"think" + 0.032*"stuff" + 0.030*"keep" + 0.018*"laugh"
69 ---> 0.171*"suppose" + 0.088*"read" + 0.045*"special" + 0.041*"mile" + 0.040*"deserve" + 0.040*"train" + 0.033*"mouth" + 0.029*"sad" + 0.027*"program" + 0.025*"vote"
70 ---> 0.109*"figure" + 0.106*"speak" + 0.103*"point" + 0.082*"damn" + 0.060*"team" + 0.046*"lead" + 0.033*"search" + 0.030*"evening" + 0.023*"finger" + 0.020*"pray"
71 ---> 0.135*"actually" + 0.128*"give" + 0.110*"mother" + 0.107*"call" + 0.087*"stand" + 0.060*"trust" + 0.060*"drop" + 0.040*"realize" + 0.029*"know" + 0.023*"think"
72 ---> 0.245*"run" + 0.172*"hell" + 0.106*"ago" + 0.048*"ball" + 0.032*"freak" + 0.032*"station" + 0.024*"memory" + 0.024*"bloody" + 0.022*"bust" + 0.022*"know"
73 ---> 0.210*"wanna" + 0.126*"forget" + 0.083*"soon" + 0.077*"daughter" + 0.043*"slow" + 0.042*"song" + 0.036*"simple" + 0.036*"birthday" + 0.033*"film" + 0.025*"know"
74 ---> 0.172*"job" + 0.146*"hard" + 0.095*"close" + 0.079*"book" + 0.067*"sign" + 0.041*"certainly" + 0.023*"bob" + 0.022*"member" + 0.021*"add" + 0.020*"left"
75 ---> 0.192*"father" + 0.086*"clear" + 0.051*"as long as" + 0.048*"mad" + 0.045*"lord" + 0.044*"boss" + 0.042*"sweet" + 0.038*"bunch" + 0.033*"invite" + 0.032*"girlfriend"
76 ---> 0.364*"leave" + 0.168*"idea" + 0.043*"company" + 0.031*"think" + 0.030*"ruin" + 0.029*"assume" + 0.021*"student" + 0.019*"source" + 0.016*"something like" + 0.013*"know"
77 ---> 0.322*"god" + 0.073*"act" + 0.062*"piece" + 0.053*"straight" + 0.042*"evidence" + 0.037*"single" + 0.027*"harry" + 0.021*"shirt" + 0.020*"weight" + 0.017*"heat"
78 ---> 0.458*"say" + 0.053*"decide" + 0.038*"to do with" + 0.035*"captain" + 0.032*"john" + 0.027*"drug" + 0.026*"air" + 0.023*"road" + 0.022*"think" + 0.021*"create"
79 ---> 0.196*"woman" + 0.109*"room" + 0.069*"free" + 0.065*"different" + 0.060*"sort" + 0.051*"dear" + 0.049*"president" + 0.028*"second" + 0.023*"ya" + 0.021*"tape"
80 ---> 0.331*"love" + 0.062*"cool" + 0.054*"expect" + 0.046*"law" + 0.043*"small" + 0.035*"tree" + 0.032*"cry" + 0.029*"honor" + 0.023*"by the way" + 0.023*"arrive"
81 ---> 0.222*"help" + 0.196*"ask" + 0.115*"hand" + 0.061*"hit" + 0.055*"murder" + 0.053*"fire" + 0.021*"question" + 0.017*"think" + 0.016*"dollar" + 0.016*"boat"
82 ---> 0.385*"mean" + 0.085*"lie" + 0.049*"know" + 0.043*"game" + 0.042*"fun" + 0.037*"fast" + 0.027*"put" + 0.026*"think" + 0.025*"lucky" + 0.025*"rule"
83 ---> 0.335*"maybe" + 0.142*"shit" + 0.082*"person" + 0.044*"wake" + 0.036*"instead" + 0.034*"know" + 0.027*"reach" + 0.026*"kiss" + 0.024*"fat" + 0.016*"danger"
84 ---> 0.182*"long" + 0.170*"live" + 0.067*"moment" + 0.047*"line" + 0.044*"allow" + 0.041*"return" + 0.029*"know" + 0.028*"especially" + 0.027*"visit" + 0.024*"max"
85 ---> 0.285*"great" + 0.139*"pretty" + 0.055*"poor" + 0.028*"attention" + 0.028*"crap" + 0.021*"heaven" + 0.020*"thing" + 0.019*"sake" + 0.017*"warm" + 0.017*"know"
86 ---> 0.586*"oh" + 0.097*"away" + 0.079*"remember" + 0.035*"finally" + 0.023*"feeling" + 0.022*"choice" + 0.021*"know" + 0.016*"voice" + 0.014*"track" + 0.008*"think"
87 ---> 0.370*"good" + 0.112*"kid" + 0.102*"friend" + 0.051*"change" + 0.049*"wife" + 0.033*"fact" + 0.029*"$" + 0.027*"know" + 0.024*"think" + 0.022*"build"
88 ---> 0.395*"find" + 0.079*"somebody" + 0.077*"worry" + 0.045*"stick" + 0.031*"lock" + 0.026*"mark" + 0.025*"know" + 0.022*"bother" + 0.022*"feed" + 0.022*"marriage"
89 ---> 0.096*"black" + 0.061*"position" + 0.057*"wall" + 0.055*"risk" + 0.046*"michael" + 0.040*"climb" + 0.037*"nick" + 0.032*"neck" + 0.029*"model" + 0.029*"national"
90 ---> 0.418*"man" + 0.127*"sir" + 0.074*"brother" + 0.039*"anybody" + 0.037*"know" + 0.032*"push" + 0.029*"agree" + 0.018*"ha" + 0.016*"l" + 0.015*"department"
91 ---> 0.090*"human" + 0.072*"weird" + 0.048*"chief" + 0.038*"public" + 0.035*"nervous" + 0.029*"freeze" + 0.029*"e" + 0.028*"mail" + 0.027*"know" + 0.026*"lab"
92 ---> 0.358*"yes" + 0.091*"bit" + 0.057*"month" + 0.054*"power" + 0.028*"situation" + 0.026*"entire" + 0.026*"quiet" + 0.024*"security" + 0.024*"camera" + 0.023*"suggest"
93 ---> 0.151*"miss" + 0.089*"phone" + 0.077*"wonder" + 0.066*"death" + 0.044*"ride" + 0.038*"mistake" + 0.033*"definitely" + 0.032*"type" + 0.028*"involve" + 0.025*"horse"
94 ---> 0.193*"you know what" + 0.123*"morning" + 0.094*"town" + 0.067*"picture" + 0.065*"perfect" + 0.040*"glass" + 0.035*"shoe" + 0.026*"fit" + 0.025*"think" + 0.024*"tom"
95 ---> 0.197*"bad" + 0.103*"break" + 0.075*"catch" + 0.063*"true" + 0.062*"sister" + 0.053*"follow" + 0.044*"shall" + 0.030*"land" + 0.021*"thing" + 0.020*"apologize"
96 ---> 0.336*"♪" + 0.057*"blow" + 0.038*"agent" + 0.034*"round" + 0.034*"park" + 0.031*"alright" + 0.027*"hole" + 0.027*"guard" + 0.026*"table" + 0.025*"clearly"
97 ---> 0.095*"young" + 0.080*"promise" + 0.074*"step" + 0.066*"finish" + 0.048*"absolutely" + 0.036*"fair" + 0.035*"fool" + 0.030*"destroy" + 0.028*"adam" + 0.026*"bomb"
98 ---> 0.078*"carry" + 0.059*"appreciate" + 0.055*"david" + 0.052*"surprise" + 0.043*"o" + 0.039*"code" + 0.037*"trick" + 0.035*"y" + 0.028*"example" + 0.026*"project"
99 ---> 0.148*"sit" + 0.111*"sound" + 0.100*"buy" + 0.074*"dream" + 0.072*"light" + 0.068*"sell" + 0.050*"hot" + 0.048*"glad" + 0.028*"smell" + 0.022*"near"

```

- well, that's what it means by "latent" - you can't easily understand what topics they belong to.
- topic 90 (0.418*"man" + 0.127*"sir" + 0.074*"brother" + 0.039*"anybody" + 0.037*"know"...) seems to be something related masculinity. companion? 

what next? what's the ideal combination for the hyper parameters?


## 13th of April


You've got to finish this today. 

What's your goal? You have two corpora to extract collocations from.
1. coca_spok
2. coca_mag

Extract collocations from those corpora with three different methods: simple count, tfidf and
That's all you have to do.



아. symbolic link를 이럴 때 쓰면 되는구나...하하.

single point of access. now, create  data directory at ~ in the server.
Then, it would be much easier to sync things.