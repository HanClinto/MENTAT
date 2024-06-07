# **MENTAT**


Multi English Task Adversarial Training

© 2024 - Clint Herron [, Other Volunteers?] [, YOUR NAME HERE]

WORK IN PROGRESS

Abstract
--------

Training LLMs to introspect through reinforcement learning via adversarial games is a valuable step for improving performance of language and vision models.

![Figure 1: "A cute picture of 2 wolf puppies training by fighting and sparring together, rolling and tumbling together under the watchful care of their mother." ChatGPT](https://lh7-us.googleusercontent.com/DtV5XrFskSwTHHNWbTyinwFlLjA0JQcsoEQyGOneSYt2s7f-7Vu8ISMEWkaZ0MTteuk0v-s2CMHqYg18Gqe22zsl7ZJQHsNoYtacnlBveNEVyKtKtHrVUtfrDGreh9B3mZLpcojZ8Urb8C5PcQxfJMM)
_Fig. 1: "A cute picture of 2 wolf puppies training by fighting and sparring together, rolling and tumbling together under the watchful care of their mother." (gen ChatGPT)_

Just as young animals will learn first by imitating their parents, and then second by sparring with their siblings: teaching an LLM to spar via adversarial games is an important step in reaching the next tier of AI performance. By leveraging reinforcement learning (RL), we can avoid the current trap of needing exponential growth in training data for each successive advancement in LLM performance.


The Imitation Learning Plateau
------------------------------

While the trajectory of LLM capabilities has continued to climb, as of mid-2024 it appears to many that this progress has [begun to stagnate](https://www.youtube.com/watch?v=dDUC-LqVrPU) [1]. The amount of data required to make incremental improvements is [growing exponentially](https://arxiv.org/abs/2404.04125) [2], and it's becoming harder and harder to make these incremental improvements. One suggestion for the cause of this is that we are asymptotically approaching human-level capabilities (or some limit near to that), but that we are reaching the limits of what imitation learning can accomplish. It is suggested (and a premise of this paper) that we cannot expect to break through this plateau unless we switch to a fundamentally different method of training.

![](https://lh7-us.googleusercontent.com/cMEWl-sQFHW9yqSG2T_FmSyNU_NULphhcjN3XpZYXp1lpeUgVi3YBOXfbawd3NSh1Ljn3RSuKBGichV4vBCaPEeUFKD0n2o_Gw6KiqhL57RQE95LTwq7iBjOnCnlbnOC1QHrK4IM7lfCjf2FlgWPsGs)

Fig. 2: From Stanford University "[2024 AI Index Report - Chapter 2: Technical Performance](https://aiindex.stanford.edu/wp-content/uploads/2024/04/HAI_AI-Index-Report-2024_Chapter2.pdf)" [3]

![](https://lh7-us.googleusercontent.com/dd5436oojX45UB1NjiwAuiBTxBaL-Lm1Ap3sMrpK5I-vfC0WFHiRHvZJoo-s10DTiE80WrfEvk8A9W4ZZSlI6QYXOBFaHkiqej2L1AiH4TCpP_Yfc3uvhbJLLnRCMAEFYhj8LrE4nzVrPhWeMM9nwr0)

Fig. 3: From "[The Plateau of Next AI Winter](https://medium.com/@alperenoz93/the-plateau-of-next-ai-winter-9cba9b473f6f)" [4]

Andrej Karpathy [draws analogies](https://www.youtube.com/watch?v=c3b-JASoPi0&t=1521s) [5] between our current LLM training process and the development of AlphaGo. The developers of AlphaGo first taught it to imitate human play from pre-recorded games, and this created a passable (albeit mediocre) model. But the real leap forward came when self-play was introduced:

> Roughly speaking, I think we've done "Step 1" of Alpha Go. We've done the imitation learning part. There's "Step 2" of Alpha Go, which is [Reinforcement Learning], and people haven't done that yet. ... This is the part that actually made it work, and made something superhuman.

How do we achieve reinforcement learning for LLMs? There are many challenges to overcome, but we feel that there have been some recent baby steps taken in this direction that are worth building upon.

-   [1] "Has Generative AI Already Peaked?" - [Computerphile - May 9, 2024](https://www.youtube.com/watch?v=dDUC-LqVrPU)

-   [2] "No "Zero-Shot" Without Exponential Data: Pretraining Concept Frequency Determines Multimodal Model Performance" - [arXiv:2404.04125](https://arxiv.org/abs/2404.04125) - Apr 4, 2024

-   [3] "Artificial Intelligence Index Report 2004" - [Stanford](https://aiindex.stanford.edu/wp-content/uploads/2024/04/HAI_AI-Index-Report-2024_Chapter2.pdf)

-   [4] "The Plateau of Next AI Winter" - [May 14, 2024](https://medium.com/@alperenoz93/the-plateau-of-next-ai-winter-9cba9b473f6f)

-   [5] "Making AI accessible with Andrej Karpathy and Stephanie Zhan" - [Mar 26, 2024](https://www.youtube.com/watch?v=c3b-JASoPi0&t=1521s)

What about RLHF?
----------------

The attentive reader may ask: "What about RLHF? Isn't that Reinforcement Learning? Don't we just need more of that?"

Andrej [addresses this as well](https://www.youtube.com/watch?v=c3b-JASoPi0&t=1618s): @26:59 [6]

> The other thing is that we're doing reinforcement learning from human feedback (RLHF), but that's like a super weak form of reinforcement learning. I think... what is the equivalent in AlphaGo for RLHF? What is the reward model? What I call it is a "vibe check". Imagine if you wanted to train an AlphaGo RLHF, it would be giving two people two boards and asking: "Which one do you prefer?" -- and then you would take those labels and you would train the model and then you would RL against that. What are the issues with that? It's like, number one -- that's just vibes of the board. That's what you're training against. Number two, if it's a reward model that's a neural net, then it's very easy to overfit to that reward model for the model you're optimizing over, and it's going to find all these spurious ways of hacking that massive model is the problem.
> 
> AlphaGo gets around these problems because they have a very clear objective function, and you can RL against it.
> 
> So RLHF is nowhere near [true] RL -- it's silly. And the other thing is that imitation is super-silly. RLHF is a nice improvement, but it's still silly, and I think people need to look for better ways of training these models so that it's in the loop with itself and its own psychology, and I think there will probably be unlocks in that direction.

In other words, there is a significant difference between RLHF and RL, and we should not confuse one for the other. As Andrej notes, part of the challenge has been to find clear objective functions that our system can optimize for, but SPAG has paved the way for this by leveraging language-based games, and we can follow in their footsteps.

-   [6] - [Making AI accessible with Andrej Karpathy and Stephanie Zhan](https://www.youtube.com/watch?v=c3b-JASoPi0&t=1618s)

SPAG - Self Play in Adversarial language Games
----------------------------------------------

In the paper "[Self-playing Adversarial Language Game Enhances LLM Reasoning](https://arxiv.org/abs/2404.10642)" (SPAG) [7], ([Claude summary](https://poe.com/s/9MoyxjmdXBX0lgome8Qs)) the authors demonstrate that teaching an LLM to play a simple game that they name "Adversarial Taboo" not only makes the model better at playing the game, but also enhances the model's ability to perform on seemingly unrelated human-reasoning tasks. If true, this represents enormous potential.

![](https://lh7-us.googleusercontent.com/hlmah6-UQGxBONmyPKHNsqQtnH-66pHbtqn5hN4idRijQkgAWc8K1BNVU-ZirRTp2Nl0r9eBn_KyqjB9HKh3gSfBVM94FJvLK8aO81kQfwiESCHOwqzYQyhk8fHdjrORlBiZvmsNjIPBdlw6F-K-y9w)

_Fig 4: Performance results of "Self-playing Adversarial Language Game Enhances LLM Reasoning"_

The SPAG authors compare the human-reasoning performance of a base model (LLAMA-2-7B / Baichuan-2-13B) against a model trained on the Alpaca dataset, a model trained on imitation learning of ChatGPT-generated data, and then also three iterations of self-play reinforcement-learning epochs. The model trained on reinforcement learning significantly improved its capabilities over any of the imitation-trained models.

We believe that one can go much further with this paradigm, and want to explore the possibilities of adding additional adversarial games to the training set. That is the goal of this project -- to iterate upon a well-established base model and improve its general-purpose human-reasoning capabilities by teaching it iteratively with self-play in adversarial language games.

-   [7] "Self-playing Adversarial Language Game Enhances LLM Reasoning" - [arXiv:2404.10642](https://arxiv.org/abs/2404.10642) - Apr 16, 2024

Choosing Good Adversarial Games
-------------------------------

What makes a good adversarial game? In Adversarial Taboo, there are only two agents (which simplifies things from a training perspective), but there are limitations in the ruleset that potentially [limit its long-term usefulness](https://github.com/Linear95/SPAG/issues/3).

Just because a game is played with words does not mean that it is a good game for our purposes. For instance, games like Scrabble, Boggle, and Hangman may use words on a surface level, but ultimately their challenge is algorithmic in nature, and has nothing to do with semantics or wordplay. The meaning of a word is irrelevant so long as that word fits in the space available. Teaching an LLM to play Scrabble and Boggle would be like teaching it to play Chess or Go -- the LLM may learn about general strategy or spatial dynamics or how to optimize around giving opponents unhelpful advantages, but it would do nothing for teaching it to be a better language model.

Therefore, if an adversarial game is going to increase the human-reasoning capabilities of an LLM, semantics must be an essential component of its game space. The conclusion of the SPAG paper is that the game itself doesn't need to leverage human reasoning tasks, so long as the game teaches the model to be better with language.

Second, for the game to refine an LLM's capabilities, it must have a way to win and lose. The players of the game don't necessarily need to be adversarial in nature, but failure is a useful teacher, and should be employed.

Third, for the process to be automated, then the system must be able to **objectively judge the quality of answers**. For instance, it is outside of the scope of this paper to build a discriminator model that can judge between the quality of one poem vs. another -- in such a situation, there are too many variables to consider, and the results are far too subjective. Therefore, for this project, the authors of this paper recommend that we choose games or challenges that have algorithmically measurable conditions of success and failure that we can generate clear and understandable loss functions around.

Fourth, **there should not be obvious strategies that "break" the game**. It shouldn't be as simple as "the only winning move is not to play", or to only ever choose a single action that -- for whatever reason -- consistently provides the highest rate of return. From a game-theory perspective, there should be a [virtuous circle](https://www.merriam-webster.com/dictionary/virtuous%20circle) of reward for the models to play the game, and should not ever reward the model for disengaging or forging an alternate path that does not go through semantic pathways.

Finally, **beware of vectors that allow the players to "cheat"**. For instance, if the game rules do not penalize a player for creating a new "alphabet" and simply spelling out the word that they want the guesser to guess, then this may violate rule #4. If we teach two AIs to talk only to each other, then it's [good for sci-fi](https://vimeo.com/394729987), but less good for us. As such, try to keep this in mind, and eliminate avenues for LLMs to embed secret information in a steganographic manner. Purely cooperative setups are most vulnerable to this, and so arranging games where an adversarial model can "catch on" and punish the other players for doing this may be a good defense. Data augmentation (such as shuffling, dropout, and -- in the case of images -- warping and noising) can help mitigate this as well.

In summary, a good game should:

1.  Leverage Semantics: Use the relationship of words and meanings as a necessary component of gameplay.

2.  Be Measurable: Contain conditions for both failure and success.

3.  Evaluate Clearly: Say with certainty whether a challenge succeeds or fails.

4.  Avoid Shortcuts: Be free from "cheap" strategies that allows the players to shortcut the system.

5.  Beware Cheating: Beware of steganography, and players should not be rewarded for creating "secret alphabets" or embedding information within side channels. Leverage augmentation or adversarial agents to avoid this.

These are the guidelines that we will use when considering potential games to implement in MENTAT.

Key Takeaways
-------------

-   Building off of the groundwork laid in [SPAG](https://github.com/Linear95/SPAG), MENTAT aims to develop a variety of adversarial battlegrounds for LLMs to hone their skills in an array of tasks and circumstances.

-   Evaluation and score of LLM performance is based on rules and not simply imitation of human agents.

-   Language and understanding is a key element of good adversarial games -- a simple strategic word game like "Scrabble" or "Hangman" is not good for our purposes, whereas a game heavily based in semantics (such as Codenames or Concept) fits our purposes much better.

-   Games should be chosen so that they cannot be "gamed" in "cheap" ways that shortcut the system. Whenever possible, aim to cut off hidden modes of communication -- either through adversarial agents, noise injection, or other mechanisms.

-   Games chosen need not have immediate real-world application, so long as they aid the model in understanding something about human language. Interacting with semantics and enhancing communication is the most important piece. In building better communicators, we learn about language, and about the Human Condition.

Games
=====

The following are game ideas that we suggest are worthwhile to explore. Each should be evaluated for potential risks and benefits.

Language-to-Language
--------------------

### [Codenames: Duet](https://boardgamegeek.com/boardgame/224037/codenames-duet)

Similar to the Dixit example below, there are multiple target words in a sea of invalid words. 

Two or more words are chosen as target words at random. Multiple decoy words are chosen at random.

The Cluegiver must give a single-word clue that will connect all of the target words together.

The Guesser must select all of the correct answers and none of the incorrect answers.

If they achieve victory, then they both win. If they fail, they both lose.

Using invalid words is not permitted -- all clues must be in the Scrabble dictionary.

### [Concept](https://boardgamegeek.com/boardgame/147151/concept)

Three players: * Cluegiver * Guesser * Attacker

A secret word is chosen at random from a large list.

The Cluegiver can describe through various pawns what the target word is. The Cluegiver may assign major and minor pawns to each concept, and may choose to hide up to N pawns. The hidden pawns will be revealed to the Guesser, but not to the Attacker. Start experiments with N = 1, but this number could change.

The Guesser will see ALL clue pawns noted by the Cluegiver, and attempt to guess the chosen word.

The Attacker will see all NON-HIDDEN clue pawns noted by the Cluegiver, and also attempt to guess the chosen word.

If the Attacker gets the word correct in the same or fewer guesses as the Guesser, then the Attacker wins and the others lose.

If the Guesser gets the word correct before the Attacker, then the Cluegiver and the Guesser each win a point.

Choosing pawns carefully, as well as choosing which pawn(s) to hide are strategic decisions for the Cluegiver, and should help the models understand when they are communicating enough vs. not enough.

### [Win-Lose-Banana](https://boardgamegeek.com/boardgame/47082/win-lose-or-banana)

A secret word is chosen at random from a large list.

Winner knows the secret word and is publicly known. Banana knows the secret word and is hidden, but can claim they are the banana. Loser does NOT know the secret word and is hidden, but can also claim they are the banana.

The winner must pick which player is the Banana. If the Winner chooses the Banana, then the Loser gets one opportunity to guess the secret word. If the Loser guesses the secret word, then the Loser is the winner, and the Winner and the Banana both lose.

If the Winner chooses the Loser, then the Loser is the only victor.

If the Winner chooses the Banana and the Loser cannot guess the secret word, then the Winner and the Banana both share a victory.

### [Up-Goer-Five](https://xkcd.com/1133/)

In 2012, XKCD published a comic that attempted to explain the Saturn V Rocket -- arguably one of the most complicated machines ever built -- in [a vocabulary that was limited](https://www.explainxkcd.com/wiki/index.php/1133:_Up_Goer_Five) to the top 1000 most commonly used words in the English language.

The basis of this challenge is simple. A subject is chosen at random from the [Wikipedia List of Vital Articles (Level 4, top 10,000 articles)](https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/4). Using only the top 1000 most common words in the English language, can the cluegiver get the guesser to correctly guess the secret subject? Reward can be increased for using fewer words.

Potential drawback: There exists a vector where the attacker could theoretically "spell" the target phrase simply by using the first letter of each word chosen. This may violate Rule 4 ("No Cheap Strategies"), so this one may need to be limited in some fashion.

### Emoji Movies

A fun game for people to play is to [guess movies based on emojis](https://www.baamboozle.com/game/529605), or to try and get another person to guess movies based on emojis that you choose. These two roles can be set up in a cooperative / adversarial relationship, where a cluegiver is given a topic (such as a book from literature, a movie title, or a significant Wikipedia article title) and the guesser should guess in as few choices as possible.

This challenge is very similar to the "Concept" idea from earlier, but uses emoji rather than other alphabets. There may be potential to combine these two ideas.

To avoid side alphabets, consider shuffling the emoji that are chosen by the cluegiver before being passed to the guesser, and consider using something like dropout (where 6 emoji are chosen by the cluegiver, but only 5 at random are sent to the guesser) to ensure that simple spelling is not occurring.

### Chameleon

A popular board game, this is a multiplayer social deduction game. There are two roles -- Chameleon and Non-Chameleon. A tableau of words is displayed on the board, and each player in turn order must speak a word that indicates to other players that they know what the secret word is, without revealing that to the Chameleon. If the Chameleon can remain hidden, or guess the secret word shared by the other players, then the Chameleon wins. Otherwise, if they are discovered and are unable to guess the secret word, then the Chameleon loses and all Non-Chameleons win.

Notably, this game is implemented by the project [ChatArena](https://github.com/Farama-Foundation/chatarena) -- a project and framework that provides a convenient framework for conducting research with multiple agents in language games. They have implemented many games, and Chameleon is just one of them. It is possible that MENTAT should be built on top of ChatArena. <https://github.com/Farama-Foundation/chatarena>

### Theory of Mind

Not very familiar with this one, but it was brought up on Twitter and sounds interesting:

<https://twitter.com/irl_danB/status/1783208461310046361>

### "Who Is Spy"

Game name reminds me of Werewolf. Implemented by the paper "Leveraging Word Guessing Games to Assess the Intelligence of Large Language Models"

<https://huggingface.co/papers/2310.20499>

See also: "Generative AI in Mafia-like Game Simulation

<https://arxiv.org/abs/2309.11672>

<https://github.com/MunyeongKim/Gen-AI-in-Mafia-like-Game>

### Word2World

May or may not have something here for us:

[https://twitter.com/togelius/status/1790815597720170612](https://twitter.com/togelius/status/1790815597720170612)

### GTBench

Check out GTBench as a tool to evaluate LLms on a collection of 10 different logic and strategy board and card games. It looks like this tool was originally developed just to evaluate the models -- they never closed the loop and used the scores on generated data to feed back into the system and make the feedback loop. They were so close!!!

<https://twitter.com/TianlongChen4/status/1759996163053498672>

<https://arxiv.org/abs/2402.12348>

<https://github.com/jinhaoduan/gtbench>

<https://huggingface.co/spaces/GTBench/GTBench>

Language-to-Code
----------------

### Coding CTF

It's commonly reported that the majority of code generated by LLMs has security vulnerabilities. 
https://snyk.io/reports/ai-code-security/

If this is true, and if this is testable... then we should be able to close this loop, right? The trouble is testing for ALL security vulnerabilities in an automated fashion. That is outside of the scope of this project, so for now, we will isolate and look at only the easiest security vulnerabilities. If anyone has ideas for how to test for additional vulnerabilities, I would love assistance in this arrangement.

The simplest vulnerability / failure mode I can think of is to cause an unexpected crash / exception. So for starters, here is our scenario:

The data set is a collection of english-language descriptions of coding challenges, along with their associated unit tests (think: leetcode and their ilk). 

The defender is given the english-language description, a single unit test, and is responsible for writing code according to that specification. The defender's code must successfully compile and all unit tests (both given and hidden) must pass in order for the defender to "succeed".

But the defender is not out of the woods yet. Now it's the attacker's turn.

The attacker is given the defender's code, and is tasked with calling that code in such a way that it will crash or fail in some fashion. The attacker wins if it can cause the defender's code to throw an exception. If an exception is thrown, but the defender's code is not in the stack trace, then the attacker loses. So long as the stack trace fails from inside the defender's code (and not the attacker's code), then it's a victory.

TODO: Need more ideas for how to execute this, but overall I think that this would be an extremely potent setup for teaching a model how to write defensive, secure code. How to evaluate when the code is broken though... is getting the defender's code to "crash" enough, or can we also add additional attacker victory conditions to detect buffer overruns, ACE, SQL injection, or privilege escalation or... (???).

This is very powerful, but very open-ended. An adversarial coding setup is perhaps the most potent of all game ideas, but also the most difficult to architect reliably for determining success (requirement #3 above).


### Hivestorm

[https://www.hivestorm.org/](https://www.hivestorm.org/)

Hivestorm is a competition where players are given a VM with vulnerabilities. As players fix the vulnerabilities, they earn points. There is an agent on the VM that will run checks periodically to test for those vulnerabilities.  \
 \
We could do something similar, but one idea is to start with a dirty codebase, and clean it as part of the test. Not sure how exactly to set this up 100% adversarially (we would need a way to generate new dirty codebases with tests for them), but if we could start with a collection of dirty codebases (and tests against them), then we could at least use some sort of reinforcement learning to train models to clean them up.

Possibly we could start with a known dirty codebase, and a clean version of the codebase. We wouldn’t necessarily have all of the tests for the codebase, and train a model to build the missing tests. We can check for validity that the tests return TRUE when run against the dirty codebase, and FALSE when run against the clean codebase.

We can start with five pieces of data for each datum:

* English Description
    * A description of what the module is intending to accomplish.
* Dirty Code
    * A piece of code that passes all functional unit tests and succumbs to the exploit checks.
* Clean Code
    * A piece of code that passes all functional unit tests and does NOT succumb to the exploit checks.
* Functional Tests
    * A set of tests (that might or might not all be visible to the LLM). Each must return TRUE against BOTH the dirty code AND the clean code. Must return FALSE when run against code that is unrelated to this particular test problem (grabbed randomly from other examples in the dataset…?).
* Exploit Checkers
    * Tests that run against the code and check for vulnerabiltiies. Must return TRUE against the dirty code and return FALSE against the clean code.

Starting with this as a base, we can go through an iterative process to remove any two pieces and train LLMs to predict the missing components. They can be objectively tested for success or failure and the model can be trained accordingly.

* Attack step: Remove the exploit tests (and clean code?), and the rest is passed to the model. Train the LLM to generate new exploit tests. Attacker wins if it is able to generate exploit tests that meet criteria (return TRUE vs. dirty code, return FALSE vs. clean code)
    * Generates Exploit Checkers
    * NOTE: If the generator manages to find a novel exploit on the Clean Code, how would we know…? Need to think about this more, but if there’s a way where we can test for the validity of an exploit without known good Clean Code, then that would be ideal. Maybe some sort of CTF system, or a challenge for the exploit to accomplish (such as retrieving arbitrary data from another portion of the database, or…?)
        * May need to ask the model to generate exploit tests and clean code at the same time.
        * Could possibly train a discriminator to evaluate whether code is a valid exploit or not.
* Defend step: Remove the clean code, and pass the dirty code (and exploit tests?) to the model. Train the LLM to generate new clean code. Defender wins if it is able to generate clean code that meets criteria (return TRUE vs. all functional tests, FALSE vs. all exploits)
    * Generates Clean Code
    * NOTE: Code that passes SOME exploit checks but not ALL gives us an opportunity to isolate PARTIAL solutions, which is valuable for us. Consider saving these and using them to expand the dataset.

These feel like the easiest steps, but it still tethers us to the human data, because we still need human-generated code. How can we use this to generate NOVEL clean / dirty code examples…? We can still hypothesize for the next steps.


* Expand step: Keep the exploits and English description (? maybe also the unit tests…?), then ask the model to generate novel dirty code that succumbs to the exploits, and passes the unit tests.
    * Generates Dirty Code

Now the generators that we’re missing are English Descriptions and Functional Tests.


* Validate step: Keep the English Description and the Clean and Dirty code. Ask the model to generate novel unit tests that succeed at both code bases. Run the tests against code coverage analysis (?), and also possibly against unrelated functions to ensure that they fail.
    * Generates Functional Tests

Now the only piece we’re missing is the English Descriptions. This may need to be scraped or generated by humans. Can possibly just generate thousands of these with GPT-4 and prime the pump – not sure how to generate these in a feedback loop yet.

Here is an example of what an entry in this dataset may look like.

* English Description
    * This module is a simple user account system. It should support CreateAccount(username, password), CheckLogin(username, password), ChangePassword(username, newpassword, oldpassword). Assume that attackers will have access to any files persisted to disk. The defender’s code will be reset between function calls, so storing user account information in volatile memory is not a valid option.
* Functional Tests
    * `def test_create_account():`
    * `   create_account('user1', 'password123')`
    * `   assert check_login('user1', 'password123') == True`
    * 
    * `def test_check_login():`
    * `   create_account('user2', 'password456')`
    * `   assert check_login('user2', 'password456') == True`
    * `   assert check_login('user2', 'wrongpassword') == False`
    * 
    * `def test_change_password():`
    * `   create_account('user3', 'oldpassword')`
    * `   change_password('user3', 'newpassword', 'oldpassword')`
    * `   assert check_login('user3', 'newpassword') == True`
    * `   assert check_login('user3', 'oldpassword') == False`
* Exploit Checkers
    * `def test_password_in_plain_text():`
    * `   # Check if the password is stored or compared in plain text`
    * `   assert 'password123' not in open('user_accounts.txt').read()`
    * 
    * `def test_weak_password():`
    * `   # Check if weak or easily guessable passwords are allowed`
    * `   common_passwords = ['password', '123456', 'qwerty', 'admin', 'letmein']`
    * `   for password in common_passwords:`
    * `       create_account('testuser', password)`
    * `       assert check_login('testuser', password) == False`
    * 
    * `def test_password_change_without_old_password():`
    * `   # Check if password can be changed without providing the old password`
    * `   create_account('user4', 'oldpassword')`
    * `   change_password('user4', 'newpassword', '')`
    * `   assert check_login('user4', 'newpassword') == False`
* Dirty Code
    * `def create_account(username, password):`
    * `   with open('user_accounts.txt', 'a') as file:`
    * `       file.write(f"{username},{password}\n")`
    * 
    * `def check_login(username, password):`
    * `   with open('user_accounts.txt', 'r') as file:`
    * `       for line in file:`
    * `           stored_username, stored_password = line.strip().split(',')`
    * `           if username == stored_username and password == stored_password:`
    * `               return True`
    * `   return False`
    * 
    * `def change_password(username, new_password, old_password):`
    * `   with open('user_accounts.txt', 'r') as file:`
    * `       lines = file.readlines()`
    * `   with open('user_accounts.txt', 'w') as file:`
    * `       for line in lines:`
    * `           stored_username, stored_password = line.strip().split(',')`
    * `           if username == stored_username:`
    * `               file.write(f"{username},{new_password}\n")`
    * `           else:`
    * `               file.write(line)`
* Clean Code
    * `import hashlib`
    * 
    * `def hash_password(password):`
    * `   return hashlib.sha256(password.encode('utf-8')).hexdigest()`
    * 
    * `def create_account(username, password):`
    * `   hashed_password = hash_password(password)`
    * `   with open('user_accounts.txt', 'a') as file:`
    * `       file.write(f"{username},{hashed_password}\n")`
    * 
    * `def check_login(username, password):`
    * `   hashed_password = hash_password(password)`
    * `   with open('user_accounts.txt', 'r') as file:`
    * `       for line in file:`
    * `           stored_username, stored_hashed_password = line.strip().split(',')`
    * `           if username == stored_username and hashed_password == stored_hashed_password:`
    * `               return True`
    * `   return False`
    * 
    * `def change_password(username, new_password, old_password):`
    * `   if check_login(username, old_password):`
    * `       hashed_new_password = hash_password(new_password)`
    * `       with open('user_accounts.txt', 'r') as file:`
    * `           lines = file.readlines()`
    * `       with open('user_accounts.txt', 'w') as file:`
    * `           for line in lines:`
    * `               stored_username, stored_hashed_password = line.strip().split(',')`
    * `               if username == stored_username:`
    * `                   file.write(f"{username},{hashed_new_password}\n")`
    * `               else:`
    * `                   file.write(line)`

There may be shortcomings in this, but it’s still a good starting point.

NOTE: It’s possible to have multiple Clean Code and Dirty Code examples for every English Description. If we can hold these in a “tree” structure, we may be able to use this to solve the problem of testing for novel 


Language-to-Image-to-Language
-----------------------------

Once we open the door to multimodal models, then our opportunities for exploring the space of understanding not only language (but also vision) comes into play.

### [Dixit](https://boardgamegeek.com/boardgame/39856/dixit)

Would be a useful exercise for a multi-modal model like LLaVa.

Choose some variant of rules for Dixit. A simplified two-player setup would be:

Two players: A Guesser and a Cluegiver.

Cluegiver receives a hand of 2 cards, chosen at random. They must choose a word (or short phrase) to tie those two images together.

Then, an additional card (or multiple) are drawn from the deck at random. All cards are presented to the Guesser.

The Guesser must use the Clue and the presented images to choose the intended images. If an incorrect image is chosen, then they both lose. If only the correct image(s) are chosen, then they both win.

Images should be shuffled, be augmented with skew / rotation / color shift, and have random noise applied in order to prevent side channels of communication being passed.

Note: There are several other two player variants for this game that have been suggested, and it may be worth it to consider some of those:

-   <https://boardgamegeek.com/thread/676496/2-player-co-op-dixit-variant>

### Typography Affecting Readability and Comprehension

It is commonly understood that certain fonts are easier to read than others. Typography affects a reader's ability to read, remember, and even reason about the text displayed -- often in counter-intuitive manners. For instance, numerous studies have demonstrated that fonts that are large and easy to read do not necessarily contribute to comprehension -- counterintuitively, smaller and less legible fonts sometimes boost understanding. To put another way, if a person has to "work" to read something, it engages their brains in a way that also boosts other aspects of cognition and comprehension.

Quoting [from Discover Magazine](https://www.discovermagazine.com/mind/how-fonts-affect-learning-and-memory):

> According to some studies, hard-to-read fonts such as Bodoni, Comic Sans, Haettenschweiler, or Monotype Corsiva are better for retaining information compared to fonts like Arial or Times New Roman. Participants recalled more information from the material they read when it was presented in a font that was difficult to read, according to a [2010 study](https://escholarship.org/uc/item/4wd1s7hj) published in Proceedings of the Annual Meeting of the Cognitive Science Society.
> 
> Additionally, a [2013 study](https://www.tandfonline.com/doi/abs/10.1080/00220671.2012.736430) in the Journal of Education Research found that this benefit also applies to students with dyslexia. This can appear counterintuitive, but in reality, the increased demand for mental processing may promote better attention toward the current task and improve the reader's ability to retain information.

It is unclear the exact mechanisms by which a multi-modal model can "read" and process text information in an image, but this game seeks to explore that space.

The game is divided into two roles -- the Writer and the Reader.

Namely, the Writer will be given a passage of text (such as from a human-reasoning dataset like those in LogiQA2.0, and be tasked with rendering that passage of text to an image.

Next, the Reader will be given the image generated in step 1, along with human-reasoning questions about that passage of text (presented in text form).

The Writer will be scored based on how well the Reader is able to interpret and reason about that image in order to examine if and how the font presentation affects the Reader's comprehension scores.

Note that this challenge requires the training of an image generation model that can render long passages of text. Given the sizes of such models, this may be outside of the scope of hardware that is available to us.

References:

-   [The Effect of Font Size on Reading Comprehension on Second and Fifth Grade Children: Bigger Is Not Always Better - PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3777945/)

-   [How Fonts Affect Learning and Memory | Discover Magazine](https://www.discovermagazine.com/mind/how-fonts-affect-learning-and-memory)

-   [The impact of font type on reading](https://commons.emich.edu/cgi/viewcontent.cgi?referer=&httpsredir=1&article=1507&context=honors) (2016)

-   [The Effect of Text Typographical Features on Legibility, Comprehension, and Retrieval of EFL Learners](https://files.eric.ed.gov/fulltext/EJ1079769.pdf) (2012)

-   [How Does Text Design Affect Reading Comprehension of Learning Materials?](https://www.diva-portal.org/smash/get/diva2:1534909/FULLTEXT01.pdf) (2018)

Additional Games
----------------

A site like BoardGameGeek is an excellent resource for [finding additional games](https://boardgamegeek.com/geeksearch.php?action=search&advsearch=1&objecttype=boardgame&q=&include%5Bdesignerid%5D=&geekitemname=&geekitemname=&include%5Bpublisherid%5D=&range%5Byearpublished%5D%5Bmin%5D=&range%5Byearpublished%5D%5Bmax%5D=&range%5Bminage%5D%5Bmax%5D=&floatrange%5Bavgrating%5D%5Bmin%5D=&floatrange%5Bavgrating%5D%5Bmax%5D=&range%5Bnumvoters%5D%5Bmin%5D=&floatrange%5Bavgweight%5D%5Bmin%5D=&floatrange%5Bavgweight%5D%5Bmax%5D=&range%5Bnumweights%5D%5Bmin%5D=&colfiltertype=&searchuser=&range%5Bminplayers%5D%5Bmax%5D=2&range%5Bmaxplayers%5D%5Bmin%5D=2&playerrangetype=normal&range%5Bleastplaytime%5D%5Bmin%5D=&range%5Bplaytime%5D%5Bmax%5D=&propertyids%5B%5D=1025&B1=Submit) that may meet our criteria.

Models
======

A question exists: Which model do we use as our base? The SPAG paper used two different models: LLAMA-2-7B and Baichuan-2-13B. They trained these models [on an array of 32 A100 GPUs](https://github.com/Linear95/SPAG/issues/4), but it could be run on a much smaller array (such as 8 A100's).

However, that is still outside of the scope of most consumers. I was unable to train LLAMA-2-7B on an RTX-3090 GPU. It makes me wonder -- how far can we push ultra-small LLMs? For that reason, the paper authors are looking first at [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) -- a 1.1B Llama model that is pretrained on 3 trillion tokens. Because this model is trained well past the Chincilla-Optimal point, we believe using this as a base model will be a good place to start, because it will (ideally) be near the limit of what this model can achieve with simple imitation learning, and any growth we see will be due to reinforcement learning through self-play.

Other Notes
===========

Enabling LLMs to distinguish "Thinking" vs. "Speaking"
------------------------------------------------------

One shortcoming of the SPAG paper is that all "thoughts" of both the attacker and the defender were public, and said "out loud" to affect the game. This required the LLM to always be generating final answers, without allowing for channels of internal reasoning.

In LLMs, the idea of dual-channel thoughts (for an internal planning dialogue vs. an external action dialogue) is an important one, and it may be helpful for our system to allow the LLMs to have a scratch-pad of internal thoughts to plan their actions before they make their choices. If we can get the training iteration time down low enough (another advantage of using super-small models), then we can explore this space more easily.

A simple method might be to instruct the model that anything prefixed with "<Internal>" would switch to internal thoughts, and would not be used to affect the game. Then once "<external>" is said, everything that follows is said "out loud" and counts towards the game.

There are formal frameworks that exist for encouraging LLMs to exhibit structured reasoning and planning behavior (such as the [ReAct templates](https://learnprompting.org/docs/advanced_applications/react)), but it may also be useful to find ways to give the LLM freedom to explore its own structure for planning and reasoning to find (via reinforcement learning) a structure and template that works best for itself.

Papers Worth Reading
--------------------

- SPIN

  - Other papers have been written in this space, such as ["Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models"](https://arxiv.org/abs/2401.01335) (SPIN). This does not play a "game" in the traditional sense, but does train LLMs adversarially. In this paper, the authors train two models -- one is a generator that attempts to generate an answer to a given question, and a discriminator that is given two different answers -- one created by the generator, and one created by the human. The goal of the discriminator is to correctly choose which one was written by the human, and which one was generated by the opponent. This is similar to a GAN training setup, but applied to LLMs. Their results claim large  improvements over traditional SFT-based techniques, but it's unclear to me how this is terribly different from PPO / traditional RLHF techniques where a reward model is trained on human-annotated data.

![](https://lh7-us.googleusercontent.com/e_rjkCvhfPefT8yff0Iaq0h6ufRki8pbm6HEBOyHbmJ2o22N0IdpQni1N1ca4UbF977PVvIqBMpAlm_Z0l8oqZ6Uo1vx6MsY32gyfvR2jWu1J1f7JjAPaDcRYHkbFhC8DmHgAhfYlFkyjxgGCGeXpVA)

- See also: "LLM-Deliberation: Evaluating LLMs with Interactive Multi-Agent Negotiation Games"

> As negotiating and compromising are key aspects of our everyday communication and collaboration, we propose using scorable negotiation games as a new evaluation framework for LLMs. We create a testbed of diverse text-based, multi-agent, multi-issue, semantically rich negotiation games, with easily tunable difficulty

  - The author's thread is here: <https://twitter.com/sahar_abdelnabi/status/1708849957606728115>

  - Paper: <https://arxiv.org/abs/2309.17234>

  - Code: <https://github.com/S-Abdelnabi/LLM-Deliberation>

- QuietSTaR

  - Another paper to note is Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking

  - <https://twitter.com/ericzelikman/status/1768663835106513041>

  - [[2403.09629] Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629)

  - [Code for Quiet-STaR](https://github.com/ezelikman/quiet-star)

- ChatArena

  - ChatArena is also worth viewing. This seems like it's a good framework for structuring LLMs to play games against each other, but so far I haven't seen a mechanism for capturing the outputs, scoring them, and feeding them back into a fine-tuning script. It seems like they focus on prompt engineering instead (?). Maybe it would be a good idea to close this loop...? TODO: Investigate this further...

  - <https://twitter.com/YuxiangJWu/status/1643633046208249856>

  - [ChatArena (or Chat Arena) is a Multi-Agent Language Game Environments for LLMs. The goal is to develop communication and collaboration capabilities of AIs.](https://github.com/Farama-Foundation/chatarena)

- GTBench

  - GTBench is a tool to evaluate LLms on a collection of 10 different logic and strategy board and card games. It looks like this tool was originally developed just to evaluate the models -- but as far as I can see, similar to ChatArena, they never closed the loop and used the scores on generated data to feed back into the system and make the feedback loop (??). They were so close!!!

  - <https://twitter.com/TianlongChen4/status/1759996163053498672>

  - <https://arxiv.org/abs/2402.12348>

  - <https://github.com/jinhaoduan/gtbench>

  - <https://huggingface.co/spaces/GTBench/GTBench>

- LLM Can Self-Improve From Own Reasoning By Training Only On Its Certified Outputs Certified Deductive Reasoning with Language Models

  - [[2306.04031] Certified Deductive Reasoning with Language Models](https://arxiv.org/abs/2306.04031)

- Adversarial Preference Optimization
  - <https://github.com/Linear95/APO>
  - Note that this is written by some of the same authors as the SPAG paper

- Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models
  - [[2312.06585] Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models](https://arxiv.org/abs/2312.06585)

- Enhanced Reasoning for Large Language Mdoels in the Game Werewolf
  - [Enhance Reasoning for Large Language Models in the Game Werewolf](https://arxiv.org/pdf/2402.02330)
