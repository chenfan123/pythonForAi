# 第一节

Welcome to this short course on LangChain for large
0:06
language model application development.
0:08
By prompting an LLM or large language model,
0:11
it is now possible to develop AI applications
0:14
much faster than ever before.
0:17
But an application can require prompting
0:19
an LLM multiple times and parsing its output, and so there's
0:23
a lot of glue code that needs to be written.
0:27
LangChain, created by Harrison Chase makes this
0:29
development process much easier. I'm thrilled to have Harrison here, who
0:34
had built this short course in collaboration with DeepLearning.ai to
0:38
teach how to use this amazing tool.
0:41
Thanks for having me. I'm really excited to be here. LangChain
0:44
started as an open source framework for building LLM
0:47
applications. It came about when I was talking to a bunch
0:50
of folks in the field who were building more
0:53
complex applications and saw some common abstractions in terms
0:56
of how they were being developed.
0:58
And we've been really thrilled at the community adoption of
1:01
LangChain so far.
1:02
And so look forward to sharing it with
1:04
everyone here and look forward to seeing what
1:06
people build with it.
1:06
And in fact, as a sign of LangChain's momentum, not
1:10
only does it have numerous users, but there are also many
1:14
hundreds of contributors to the open source, and this has been
1:18
instrumental for its rapid rate of development. This
1:21
team really ships code and features at an amazing pace.
1:25
So hopefully, after this short course, you'll
1:28
be able to quickly put together some really cool
1:31
applications using LangChain, and who knows, maybe
1:34
you even decide to contribute back to the
1:37
open source LangChain effort.
1:38
LangChain is an open source development
1:41
framework for building LLM applications.
1:44
We have two different packages, a Python one and a JavaScript one.
1:48
They're focused on composition and modularity. So
1:50
they have a lot of individual components that
1:52
can be used in conjunction with each other
1:55
or by themselves. And so that's one of the key value adds. And then
1:59
the other key value add is a bunch of different use cases. So chains
2:04
of ways of combining these modular components into
2:06
more end-to-end applications making it very easy to get
2:09
started with those use cases. In this class, we'll
2:11
cover the common components of LangChain. So we'll talk about models.
2:15
We'll talk about prompts, which are how you get models to
2:18
do useful and interesting things.
2:20
We'll talk about indexes, which are ways of ingesting
2:23
data so that you can combine it with models. And
2:26
then we'll talk about chains, which are more end-to-end use cases along
2:30
with agents, which are a very exciting type of end-to-end use case which uses
2:34
the model as a reasoning engine.
2:36
We're also grateful to Ankush Gola, who is a co-founder of LangChain alongside Harrison
2:41
Chase, for also putting a lot of thought into
2:44
these materials and helping with the
2:47
creation of this short course.
2:49
And on the DeepLearning.AI side, Geoff Ludwig, Eddy Shyu, and
2:53
Diala Ezzeddine have also contributed to
2:55
these materials.
2:58
And so with that, let's go on to the next video, where we'll learn
3:02
about LangChain's models, prompts, and parsers.

# 第二节

In lesson one, we'll be covering models, prompts, and
0:06
parsers. So models refers to the language models underpinning a
0:10
lot of it. Prompts refers to
0:13
the style of creating inputs to pass into the models.
0:16
And then parsers is on the opposite end.
0:18
It involves taking the output of these models
0:20
and parsing it into a more structured format
0:23
so that you can do things downstream with it.
0:25
Yep, so when you build an application using LLM, there'll
0:29
often be reusable models. We repeatedly prompt a model, parses
0:32
outputs, and so LangChain gives an easy set of abstractions to
0:36
do this type of operation.
0:38
So with that, let's jump in and take a look at models, prompts,
0:42
and parsers.
0:43
So to get started, here's a little bit of starter code. I'm going
0:47
to import OS, import OpenAI, and load my OpenAI secret key. The OpenAI
0:52
library is already installed in my Jupyter Notebook environment.
0:55
If you're running this locally, and you don't
0:57
have OpenAI installed yet, you might need to
1:00
run that. Bang, pip install OpenAI, but I'm not going to do
1:04
that here.
1:05
And then here's a helper function. This is actually very similar to the
1:10
helper function that you might have seen in the ChaiGPT prompt engineering
1:15
for developers course that I offered together with OpenAI's
1:18
Iza Fulford. And so, with this helper function, you can
1:22
say get completion on what is 1 plus 1, and this
1:26
will call ChatGPT, or technically the model, GPT 3.5 Turbo, to
1:30
give you an answer back like this.
1:32
1:34
Now, to motivate the line chain abstractions
1:37
for model prompts and parsers, let's say you get an email
1:42
from a customer in a language other than English.
1:47
In order to make sure this is accessible,
1:50
the other language I'm going to use is the English pirate language, where
1:54
the consists are, I'd be fuming that me blender
1:57
lid flew off and splattered my kitchen walls
1:59
with smoothie.
2:01
And to make matters worse, the warranty don't cover the cost of
2:03
cleaning up me kitchen. I need your help right now, matey.
2:07
And so, what we will do is, ask this LLM to
2:11
translate the text to American English in a
2:14
calm and respectful tone. So I'm going to set style
2:17
to American English in a calm and respectful tone. And
2:21
so, in order to actually accomplish this, if
2:24
you've seen a little bit of prompting before,
2:28
I'm going to specify the prompt using an f-string with
2:31
the instructions, translate the text that is
2:33
delimited by triple backticks into style that is style,
2:36
and then plug in these two styles.
2:38
And so, this generates a prompt that says translate the
2:42
text and so on. I encourage you to pause
2:45
the video and run the code, and also try modifying the
2:49
prompt to see if you can get a different output.
2:53
You can then
2:56
prompt the Large Language Model to get a response. Let's
2:59
see what the response is. Says translated the English pirate's message
3:03
into this very polite, I'm really frustrated that my blender lid
3:06
flew off and made a mess of my kitchen walls with smoothie
3:10
and so on.
3:12
I could really use your help right now, my friend.
3:14
That sounds very nice.
3:17
So, if you have different customers writing
3:20
reviews in different languages, not just English pirate, but French,
3:24
German, Japanese, and so on, you can imagine having to generate
3:29
a whole sequence of prompts to generate such translations. Let's
3:34
look at how we can do this in a more convenient, way
3:39
using LangChain. I'm going to import chat OpenAI. This is
3:44
LangChain's abstraction for the chatGPT API endpoint.
3:47
3:48
And so, if I then set chat equals chat OpenAI
3:51
and look at what chat is, it creates this object as follows
3:56
that uses the chatGPT model, which is also called GPT 3.5 turbo.
4:00
4:01
When I'm building applications, one thing I will often do
4:04
is set the temperature parameter to be equal to zero. So
4:08
the default temperature is 0.7.
4:10
But let me actually redo that with temperature
4:16
equals 0.0, and now the temperature is set to 0 to make this output a
4:21
little bit less random.
4:22
And now, let me define the template string as follows.
4:27
Translate the text delimited by triple vectors into
4:32
style that is style, and then here's the text. And
4:37
to repeatedly reuse this template, let's import
4:41
LangChain's chat prompt template,
4:45
and then, let me create a prompt template
4:49
using that template string that we just wrote above.
4:59
From the prompt template, you can actually
5:02
extract the original prompt, and it realizes that this
5:06
prompt has two input variables, the style and the text,
5:11
which were shown here with the curly braces.
5:15
And here is the original template as well
5:17
that we had specified.
5:19
In fact, if I print this out, it realizes it has two input variables,
5:23
style and text. Now, let's specify the style. This is a style that
5:28
I want the customer message to be translated
5:31
to, so I'm going to call this customer style,
5:34
5:35
and here's my
5:39
same customer email
5:41
as before. And now,
5:48
if I create customer messages, his will generate the prompt,
5:54
and will pass this large language model in
5:56
a minute to get a response. So if you want to look at the types,
6:00
the customer message is actually a list,
6:03
and if you look at the first element of the list,
6:09
this is more or less that prompt that you would
6:13
expect this to be creating.
6:16
Lastly, let's pass this prompt to the LLM, so I'm going
6:20
to call chat, which we had set earlier,
6:23
as a reference to the OpenAI chatGPT endpoint,
6:27
and, if we print out the customer responses content,
6:33
then, it gives you back this text translated from English
6:40
pirate to polite American English.
6:43
And of course, you can imagine other use cases where the
6:47
customer emails are in other languages and this
6:50
too can be used to translate the messages
6:53
for an English-speaking to understand and reply to.
6:57
I encourage you to pause the video and
6:59
run the code and also
7:00
try modifying the prompt to see if you can get a different
7:04
output.
7:05
Now let's hope our customer service agent replies
7:07
to the customer in their original language. So let's say,
7:11
English-speaking customer service agent writes
7:13
this and says, "Hey there customer, warranty
7:15
does not cover cleaning expenses for
7:17
your kitchen because it's your fault. You misused
7:19
your blender by forgetting to put on the lid. Tough luck.
7:22
See ya."
7:23
Now they're polite message, but
7:25
let's say this is what a customer service agent wants.
7:31
We are going to specify
7:34
that the service message is going to be
7:36
translated to this pirate style. So we want it to
7:39
be in a polite tone that speaks in
7:42
English pirate.
7:44
And because we previously created that prompt template,
7:46
the cool thing is, we can now reuse that prompt template and
7:50
specify that the output style we want is
7:52
this service style pirate
7:54
and the text is this service reply.
7:57
And if we do that,
8:01
that's the prompt.
8:04
And if we prompt,
8:07
ChaiGPT,
8:09
this is the response it gives us back. "Ahoy there, matey!
8:12
I must kindly inform you
8:15
that the warranty be not covering the expenses
8:17
or cleaning your galley."
8:19
And so on. Aye, tough luck. Farewell, me hearty.
8:22
So, you might be wondering, why are we using
8:25
prompt templates instead of, you know, just an f-string? The
8:28
answer is that as you build sophisticated applications,
8:31
prompts can be quite long and detailed.
8:33
8:34
And so, prompt templates are a useful abstraction to help you
8:38
reuse good prompts when you can.
8:41
This is an example of a relatively long
8:44
prompt to grade a student's submission
8:47
for an online learning application.
8:50
And a prompt like this can be quite long, in which you can ask the
8:54
LLM to first solve the problem, and then have the
8:57
output in a certain format, and the output in a certain format.
9:00
9:01
And wrapping this in a LangChain prompt makes
9:04
it easier to reuse a prompt like this.
9:07
Also, you see later that LangChain provides
9:11
prompts for some common operations, such as summarization, or
9:16
question answering, or connecting to SQL databases,
9:18
or connecting to different APIs. And so by using some of LangChain's
9:23
built-in prompts, you can quickly get an application working without
9:27
needing to,
9:28
engineer your own prompts.
9:30
One other aspect of LangChain's prompt libraries
9:35
is that it also supports output parsing, which we'll get to in a minute.
9:41
But when you're building a complex application using an LLM,
9:45
you often instruct the LLM
9:48
to generate its output in a certain format,
9:51
such as using specific keywords. This example on
9:54
the left illustrates using an LLM to carry out something called chain
9:59
of thought reasoning
10:01
using a framework called the React framework. But don't
10:04
worry about the technical details, but the keys of that is that
10:09
the thought is what the LLM
10:13
is thinking,
10:14
because by giving an LLM space to think, it can often get
10:18
to more accurate conclusions.
10:20
Then action as a keyword to carry the specific action,
10:24
and then observation to show what it learned from that action,
10:28
and so on.
10:30
And if you have a prompt that instructs the LLM
10:34
to use these specific keywords, thought, action, and observation,
10:38
then this prompt can be coupled with a parser
10:41
to extract out the text that has been
10:44
tagged with these specific keywords. And so that together
10:48
gives a very nice abstraction to specify the
10:51
input to an LLM,
10:53
and then also have a parser correctly interpret
10:57
the output that the LLM gives. And so with that, let's
11:02
return to see an example of an output parser using LangChain.
11:07
In this example, let's take a look at how you can have
11:10
an LLM output JSON,
11:13
and use LangChain to parse that output.
11:16
And the running example that I'll use will be
11:19
to extract
11:21
information from a product review,
11:23
and format that output in a JSON format.
11:27
So here's an example of how you would like the output to be formatted.
11:32
11:32
Technically, this is a Python dictionary,
11:35
where whether or not the product is a gift,
11:37
maps to false, the number of days it
11:39
took to deliver was five, and the price value was pretty
11:42
affordable.
11:43
So this is one example of a desired output,
11:47
here is an example
11:48
of customer review, as well as a
11:52
template to try to get to that JSON output.
11:56
So here's a customer review. It says, this
11:58
lead blower is pretty amazing. It has four settings, candle blower, gender
12:01
breeze, windy city, and tornado.
12:03
It arrived in two days, just in time for my wife's
12:06
anniversary present. I think my wife liked it
12:08
so much she was speechless. So far, I've been
12:10
the only one using it, and so on. And here's a review
12:13
template. For the following text, extract the following
12:15
information.
12:17
Specify, was this a gift? So in this case, it would be yes,
12:20
because this is a gift.
12:22
And also, delivery days. How long did it take to deliver?
12:25
It looks like in this case, it arrived in two days.
12:28
And what's the price value? You know, slightly more expensive than
12:32
the lead blowers, and so on. So, the
12:36
review template asks the LLM
12:38
to take as input a customer review
12:41
and extract
12:43
these three fields and then format the output as JSON
12:48
with the following keys.
12:55
All right. So here's how you can wrap this in
12:58
LangChain. Let's import the chat prompt template. We'd
13:01
actually imported this already earlier. So technically,
13:04
this line is redundant, but I'll just import
13:06
it again and then have the prompt templates
13:09
created from
13:10
the review template
13:13
up on top.
13:14
And so, here's the prompt template.
13:19
And now, similar to our earlier usage of a prompt template, let's
13:22
create the
13:25
messages to pass to the
13:27
OpenAI,
13:28
endpoint.
13:31
Create the OpenAI endpoint, call that endpoint, and then let's
13:35
print out the response.
13:38
I encourage you to pause the video and run the code.
13:44
And there it is. It says, give us true, delivery day is 2,
13:47
and the price value also looks
13:50
pretty accurate. But note that if we
13:55
check the type
13:59
of the response, this is actually a string.
14:02
So it looks like JSON and looks like it has key-value
14:05
pairs, but it's actually not a dictionary. This is
14:08
just
14:09
one long string.
14:10
So what I'd really like to do is go to the
14:13
response content and get the value from the
14:14
gift key which should be true, but I run this, this
14:17
should generate an error
14:18
because, well, this is actually a string.
14:21
This is not a Python dictionary. So, let's
14:25
see how we will use LangChain's parser in order to do this. I'm going to
14:33
import response schema and structured output parser from
14:37
LangChain.
14:38
And I'm going to tell it what I wanted to parse by
14:42
specifying these response schemas. So the gif schema is named gif, and
14:46
here's the description. Was the item purchased as a gift
14:49
for someone else?
14:51
Answer true or yes, false if not or unknown, and so on.
14:55
So, have a gift schema, delivery date schema,
14:57
price value schema, and then let's put all three of them into a list
15:02
as follows.
15:04
Now that I've specified the schema for these LangChain can actually
15:08
give you the prompt
15:10
itself by
15:12
having the output parser tell you what instructions it wants you
15:17
to send to the LLM. So if I were to print
15:20
format instructions.
15:23
She has a pretty precise set of instructions for the LLM that
15:26
will cause it to generate an output that the output parser
15:30
can process.
15:33
So here's the new review template,
15:36
and the review template includes the
15:38
format instructions that LangChain
15:41
generated.
15:44
And so we can create a prompt
15:47
from the review template too,
15:49
and then create the
15:52
messages that will pass
15:54
to the OpenAI endpoint.
15:57
If you want, you can take a look at the actual prompt,
16:01
which gives the instructions to extract
16:05
the fields gift, delivery days, price value,
16:09
here's the text,
16:10
and then here are the formatting instructions.
16:16
Finally, if we
16:19
call the OpenAI
16:22
endpoint,
16:24
let's take a look at what response we got.
16:28
It is now this,
16:32
and now if we use the output parser that we created
16:39
earlier,
16:40
you can then parse this into an output dictionary,
16:45
which if I print, looks like this.
16:47
And notice that this is
16:52
of type dictionary, not a string,
16:56
which is why I can
16:57
now extract the value associated with the key
17:00
gift and get true, or
17:03
the value associated with delivery days
17:06
and get two,
17:07
or you can also extract the value associated with price value.
17:13
So this is a nifty way to take your LLM output and parse it
17:18
into a Python dictionary,
17:20
to make the output easier to use in downstream processing.
17:23
I encourage you to pause the video and run the code.
17:27
And so, that's it for models, prompts, and parsers. With
17:30
these tools, hopefully you'll be able to
17:33
reuse your own prompt templates easily, share prompt templates
17:36
with others that you're collaborating with, even
17:39
use LangChain's built-in prompt templates, which as you
17:42
just saw, can often be coupled with an output
17:45
parser,
17:46
so that the input prompt to output
17:49
in a specific format and then the parser,
17:52
parses that output to store the data in a Python dictionary or
17:56
some other data structure that makes it easy for
17:59
downstream processing. I hope you find this
18:02
useful in many of your applications.
18:04
And with that, let's go into the next video where we'll see
18:09
how LangChain can help you build better chatbots, or have
18:13
an LLM have more effective chats by better managing
18:16
what it remembers from the conversation
18:18
you've had so far.

# 第三节

When you interact with these models, naturally they don't
0:06
remember what you say before or any of
0:08
the previous conversation, which is an issue when you're building
0:11
some applications like Chatbot and you
0:13
want to have a conversation with them.
0:16
And so, in this section we'll cover memory, which is basically how
0:20
do you remember previous parts of the conversation and feed that into
0:24
the language model so that they can have this
0:27
conversational flow as you're interacting with them.
0:29
So, LangChain offers multiple sophisticated options of
0:32
managing these memories. Let's jump in and take a look.
0:37
So, let me start off by importing my
0:41
OpenAI API key,
0:43
and then let me import a few tools that I'll need.
0:46
Let's use as the motivating example for memory, using
0:50
LangChain to manage a chat or a chatbot conversation.
0:54
So, to do that, I'm going to set the LLM
0:57
as a chat interface of
0:59
OpenAI with temperature equals 0,
1:01
and I'm going to use
1:04
the memory as a ConversationBufferMemory,
1:08
and you'll see later what this means.
1:11
And I'm going to build a conversation chain. Again,
1:14
later in this short course, Harrison will dive much more
1:17
deeply into what exactly is a chain in LangChain. So, don't
1:21
worry too much about the details of the syntax for now,
1:25
but this builds an LLM.
1:27
And if I start to have a conversation, conversation.predict, give
1:31
the input,
1:33
Hi, my name is Andrew.
1:36
Let's see what it says.
1:37
Hello, Andrew, it's nice to meet you. Right? And
1:39
so on. And then,
1:40
let's say I ask it,
1:43
what is 1 plus 1?
1:45
1 plus 1 is 2, and then
1:49
ask it again, you know, what's my name?
1:51
Your name is Andrew, as you mentioned earlier. Hmm,
1:53
there's a little bit of trace of sarcasm there,
1:56
not sure.
1:57
And so if you want, you can change this
2:00
verbose variable to true,
2:02
to see what LangChain is actually doing.
2:04
When you run predict, Hi my name is Andrew,
2:07
this is the prompt that LangChain is generating.
2:09
It says,
2:10
the following is a friendly conversation between a
2:12
human and an AI. AI is talkative, and so on. So this is a
2:16
prompt that LangChain has generated
2:17
to have the system have a hopeful and friendly conversation,
2:21
and it has to save the conversation, and here's the response.
2:24
And when you
2:27
execute this on the
2:30
second and third parts of the conversations,
2:32
it keeps the prompt as follows. And notice that by the time I'm
2:36
uttering, what is my name? This is the third turn, that's my
2:40
third input.
2:41
It has stored the current conversation
2:45
as follows. Hi, my name is Andrew, what is 1 plus 1,
2:48
and so on. And so this memory or this history of
2:52
the conversation gets longer and longer. In fact, up on top,
2:56
I have used the memory variable to store the memory.
3:00
So if I were to print memory.buffer,
3:03
it has stored
3:05
the conversation so far.
3:07
You can also print this out, memory.loadMemoryVariables. The
3:10
curly braces here is actually an empty dictionary. There's
3:13
some more advanced features that you can use with
3:17
a more sophisticated input, but we won't talk about
3:20
them in this short course. So don't worry about why there's an empty curly braces
3:26
here.
3:27
But this is what LangChain has remembered in the memory of the
3:31
conversation so far. It's just everything that the AI or
3:34
that the human has said.
3:37
I encourage you to pause the video and run the code.
3:40
So, the way that LangChain is storing
3:42
the conversation is with this
3:44
ConversationBufferMemory.
3:48
If I were to use the ConversationBufferMemory
3:50
to specify a couple of
3:52
inputs and outputs, this is how you add
3:54
new things to the memory if you wish to do so explicitly.
3:57
Memory.saveContext says, hi, what's up?
4:01
I know this is not the most exciting conversation,
4:04
but I wanted to have a short example.
4:07
And with that, this what the status of the memory is.
4:12
And once again, let me actually show the
4:17
memory variables. Now, if you want to add additional
4:24
data to the memory, you can keep on saving additional context. So,
4:28
conversation goes on, not much, just hanging, cool.
4:32
And if you print out the memory, you know, there's
4:34
now more stuff in it.
4:37
So when you use a large language model
4:40
for a chat conversation, the large language model
4:43
itself is actually stateless. The language model itself does not
4:46
remember the conversation you've had so far. And each transaction, each
4:51
call to the API endpoint is independent.
4:54
And chatbots appear to have memory
4:56
only because there's
4:58
usually
4:59
rapid code that provides the full conversation that's
5:02
been had so far as context to the LLM.
5:06
And so, the memory can store explicitly
5:10
the terms or the utterances so far. Hi, my name is Andrew. Hello, it's
5:13
just nice to meet you and so on.
5:15
And this memory storage is used as input
5:18
or additional context to the LLM
5:20
so that they can generate an output as if it's
5:23
just having the next
5:24
conversational turn,
5:26
knowing what's been said before.
5:28
And
5:30
as the conversation becomes long, the amounts of
5:32
memory needed becomes really, really long and does the
5:35
cost of sending a lot of tokens to the LLM,
5:39
which usually charges based on the number of tokens it needs to
5:42
process, will also become more expensive.
5:45
So LangChain provides several
5:47
convenient kinds of memory to store and accumulate the conversation.
5:53
So far, we've been looking at the ConversationBufferMemory.
5:56
Let's look at a different type of memory.
5:59
I'm going to import the conversation buffer window
6:03
memory that only keeps a window of memory. If I set memory
6:08
to conversational buffer window memory with k equals one,
6:13
the variable k equals one specifies that I wanted to remember just
6:17
one conversational exchange. That is one utterance from
6:20
me and one utterance from
6:22
chatbot. So,
6:24
now if I were to have it save context, hi, what's up, not much,
6:28
just hanging.
6:30
If I were to
6:31
look at memory.load variables, it only remembers the most recent utterance.
6:36
Notice it's dropped. Hi, what's up? It's just saying
6:40
human says not much, just hanging, and the AI says cool. So that's
6:44
because k was equal to one.
6:47
So this is a nice feature because it
6:49
lets you keep track of just the most
6:52
recent few conversational terms. In practice, you probably won't
6:55
use this with k equals one. You use this with k set
6:59
to a larger number.
7:02
But still, this prevents the memory from growing
7:05
without limit as the conversation goes longer.
7:09
And so if I were to
7:11
rerun the conversation that we had just now,
7:17
we'll say, hi, my name is Andrew.
7:21
What is 1 plus 1?
7:26
And now I ask it,
7:27
what is my name?
7:30
Because k equals 1, it only remembers the
7:32
last exchange versus what is 1 plus 1?
7:35
The answer is 1 plus 1 equals 2, and it's forgotten this early exchange
7:39
which is now, now says, sorry, don't have
7:41
access to that information.
7:43
One thing I hope you will do is pause the video,
7:47
change this to true in the code on the left,
7:51
and rerun this conversation with verbose equals true,
7:55
and then you will see the prompts actually
7:58
used to generate this, and hopefully you'll
8:00
see that the memory,
8:03
when you're calling the LLM on what is my name,
8:06
that the memory has dropped this exchange where
8:08
it learned what is my name, which is why it now says it doesn't
8:12
know what is my name.
8:15
With the conversational token
8:18
buffer memory,
8:22
the memory will limit the
8:24
number of tokens saved. And because a lot of
8:28
LLM pricing is based on
8:30
tokens, this maps more directly
8:33
to the cost of the LLM calls. So,
8:39
if I were to say the max token limit is equal to 50,
8:42
and actually let me inject a few comments. So let's
8:46
say the conversation is, AI is what? Amazing.
8:50
Backpropagation is what? Beautiful. Chatbot is what? Charming.
8:52
I use ABC as the first letter of all of these conversational terms.
8:57
We can keep track of what was said when.
9:01
If I run this with a high token limit,
9:04
it has almost the whole conversation.
9:07
If I increase the token limit to
9:10
100,
9:11
it now has the whole conversation, sign of AI is what?
9:15
If I decrease it,
9:17
Then, you know, it chops off the earlier parts
9:21
of this conversation to retain the number of tokens
9:24
corresponding to the most recent exchanges
9:27
but subject to not exceeding the token limit.
9:30
And in case you're wondering why we needed to specify an LLM, it's because
9:34
different LLMs use different ways of counting tokens, so
9:37
this tells it to use the
9:39
way of counting tokens that the chatOpenAI
9:43
LLM uses.
9:45
I encourage you to pause the video and run the code,
9:48
and also try modifying the prompt to see if you
9:51
can get a different output.
9:53
Finally, there's one last type of memory I want to illustrate here, which
9:57
is the conversation
9:58
summary
10:00
buffer
10:02
memory.
10:03
And the idea is, instead of limiting the memory to a fixed number
10:07
of tokens based on the most recent utterances or a fixed number
10:12
of conversational exchanges,
10:13
let's use an LLM
10:15
to write a summary of the conversation so far,
10:19
and let that
10:20
be the memory.
10:23
So here's an example where I'm going to create a long
10:25
string with someone's schedule, you know, there's meeting at 8am
10:28
with your product team, you need your PowerPoint presentation, and so on and
10:31
so on. So there's a
10:33
long string saying what's your schedule, you know, maybe
10:36
ending with a noon lunch at the Italian
10:39
restaurant with a customer, bring your laptop, show
10:42
the latest LLM demo.
10:45
And so, let me use a
10:48
conversation summary buffer memory
10:51
with a max token limit of 400 in this case,
10:54
pretty high token limit,
10:56
and I'm going to
10:58
insert in
11:01
a few conversational terms in which we start with hello, what's
11:04
up, no one's just hanging,
11:06
cool.
11:09
And then
11:10
what is on the schedule today, and the
11:13
response is, you know, that long schedule.
11:16
So this
11:17
memory
11:18
now has quite a lot of text in it. In fact, let's take a look
11:23
at the memory variables.
11:24
It contains that entire
11:28
piece of text,
11:30
because 400 tokens was sufficient
11:32
to store
11:34
all this text.
11:36
But now,
11:37
if I were to reduce the max token limit, say to 100 tokens,
11:41
remember, this stores the entire conversational history.
11:44
If I reduce the number of tokens
11:47
to 100,
11:49
then,
11:52
the conversation summary buffer memory has actually used an
11:55
LLM, the OpenAI endpoint in this case,
11:57
because that's where we set the LLM to, to
12:00
actually generate a summary
12:02
of the conversation so far. So the summary is, human-AI
12:05
engage in small talk before the start of the schedule, and
12:08
informs human in the morning meeting, blah, blah, blah,
12:11
lunch meeting with customer interested in AI and latest
12:14
AI developments.
12:17
And so,
12:18
if we were to have a conversation,
12:22
using this LLM,
12:27
let me create a conversation chain,
12:30
same as before.
12:33
And let's say that we were to ask,
12:35
you know, input what would be a good demo to show.
12:39
I said verbose equals true, so here's the prompt.
12:43
The LLM thinks the current conversation
12:47
has had
12:49
this discussion so far,
12:51
because that's the summary of the conversation.
12:54
And just one note, if you're familiar with the OpenAI
12:59
chat API endpoint, there is a specific system message.
13:05
In this example,
13:06
this is not using the official OpenAI system message,
13:09
it's just including it as part of the prompt here,
13:13
but it nonetheless works pretty well. And given this prompt, you know,
13:18
the LLM outputs basic constituent AI developments,
13:21
so it's just showcasing our latest NLP capabilities. Okay, that's
13:25
cool.
13:27
Well, it's, you know, making some suggestions to the cool demos,
13:30
and makes you think if I was meeting a customer, I
13:32
would say,
13:33
boy, if only there were open source framework available to
13:38
help me build cool NLP applications using LLMs.
13:42
Good things are launching.
13:46
And the interesting thing is, if you now look at
13:51
what has happened to the memory,
13:56
so, notice that here, it has incorporated the
13:59
most recent AI system output, whereas my utterance asking it,
14:03
what would be a good demo to show, has been incorporated
14:07
into the system message, you know, the overall summary of
14:11
the conversation so far. With the conversation
14:13
summary buffer memory, what it tries to do is keep the
14:17
explicit storage of the messages up to the
14:20
number of tokens we have specified as a limit. So,
14:24
you know, this part, the explicit storage, we're
14:27
trying to cap at 100 tokens, because that's what we're asked for.
14:32
14:33
And then anything beyond that,
14:34
it will use CLM to generate a summary, which is what is seen up here.
14:39
And even though I've illustrated these different memories
14:42
using a chat as a running example,
14:45
these memories are useful for other applications too,
14:48
where you might keep on getting new snippets of text,
14:51
or keep on getting new information,
14:53
such as if your system repeatedly goes online
14:56
to search for facts, but you want to keep the total memory
14:59
used to store this growing list of facts as, you know,
15:03
capped and not growing arbitrarily long.
15:05
I encourage you to pause the video and run the code.
15:10
In this video, you saw a few types of memory
15:13
including buffer memories that limits based on number of
15:17
conversation exchanges or tokens
15:20
or a memory that can summarize tokens above a certain
15:24
limit.
15:25
LangChain actually supports additional memory types as well.
15:28
One of the most powerful is vector data memory,
15:31
if you're familiar with word embeddings and text embeddings, the
15:35
vector database actually stores such embeddings.
15:37
If you don't know what that means, don't worry about it. Harrison will
15:40
explain it later.
15:42
And it can then retrieve the most relevant
15:44
blocks of text using this type of a vector database for its memory.
15:50
And LangChain also supports entity memories,
15:52
which is applicable when you wanted to remember
15:56
details about specific people, specific other entities,
15:59
such as if you talk about
16:01
a specific friend, you can have LangChain
16:04
remember facts about that friend, which would be
16:08
an entity in an explicit way. When you're implementing applications
16:12
using LangChain, you can also use multiple types
16:14
of memories such as
16:17
using one of the types of conversation memory that you saw in
16:21
this video plus additionally entity memory
16:23
to recall individuals. So this way you can remember maybe a summary
16:27
of the conversation plus an explicit way of storing
16:30
important facts about important people in
16:32
the conversation.
16:34
And of course, in addition to using these memory types,
16:36
it's also not uncommon for developers to
16:38
store the entire conversation in the conventional database. Some
16:41
sort of key value store or SQL database. So you could refer back
16:45
to the whole conversation for auditing or for
16:48
improving the system further.
16:50
And so, that's memory types. I hope you find this useful
16:53
building your own applications.
16:55
And now let's go on to the next video to learn about the
17:00
key building block of LangChain, namely the chain.

# 第四节

In this lesson, Harrison will teach the most important
0:06
key building block of LangChain, namely, the chain.
0:11
The chain usually combines an LLM, large language model,
0:14
together with a prompt, and with this building block you
0:17
can also put a bunch of these building blocks
0:20
together to carry out a sequence of operations on your text or
0:24
on your other data.
0:25
I'm excited to dive into it. Alright, to start, we're going to load the
0:30
environment variables as we have before.
0:33
And then we're also going to load some data that we're going
0:36
to use. So part of the power of these
0:39
chains is that you can run them over many inputs
0:42
at a time. Here we're going to load a pandas DataFrame.
0:45
0:45
A pandas DataFrame is just a data structure
0:48
that contains a bunch of different elements of data.
0:50
If you're not familiar with pandas, don't worry about it.
0:53
The main point here is that we're loading some data that we can then
0:56
use later on. And so if we look inside this pandas DataFrame,
0:59
we can see that there is a product column and then a review column.
1:02
And each of these rows is a different
1:04
data point that we can start passing through
1:06
our chains.
1:08
So the first chain we're going to cover is the LLM chain. And
1:11
this is a simple but really powerful chain
1:13
that underpins a lot of the chains that we'll go
1:15
over in the future.
1:17
And so, we're going to import three different things. We're going
1:20
to import the OpenAI model, so the LLM. We're going to import the chat prompt
1:24
template. And so this is the prompt. And then we're
1:27
going to import the LLM chain.
1:29
And so first, what we're going to do is we're going
1:32
to initialize the language model that we want
1:34
to use. So we're going to initialize the
1:37
chat OpenAI with a high temperature so that we can get
1:41
some fun descriptions.
1:43
Now we're going to initialize a prompt. And
1:45
this prompt is going to take in a variable called product. It's going
1:49
to ask the LLM to generate what the best name is to describe a company that
1:53
makes that product. And then finally, we're going to
1:55
combine these two things into a chain.
1:58
And so, this is what we call an LLM chain.
2:01
And it's quite simple. It's just the combination of the LLM
2:04
and the prompt. But now this chain will let us run through the
2:08
prompt and into the LLM in a sequential manner. And so
2:11
if we have a product called queen-size-sheet-set, we can run this through
2:15
the chain by using chain.run.
2:17
And what this will, do is it will format
2:19
the prompt under the hood, and then it will pass
2:21
the whole prompt into the LLM. And so we can see that we get
2:25
back the name of this hypothetical company called
2:27
Royal Bettings. And so here would be a good time to pause. And you can
2:31
input any product descriptions that you would want.
2:33
And you can see what the chain will output as a result.
2:36
So the LLM chain is the most basic type of chain.
2:40
And that's going to be used a lot in the future. And
2:43
so we can see how this will be used in
2:45
the next type of chain, which will be sequential chains. And
2:48
so sequential chains run a sequence of chains
2:50
one after another.
2:51
So to start, you're going to import the simple sequential chain.
2:56
And this works well when we have subchains
2:58
that expect only one input and return only one output.
3:01
And so here we're going to first create one chain,
3:04
which uses an LLM
3:06
and a prompt.
3:07
And this prompt is going to take in
3:10
the product and will return the best name
3:12
to describe that company. So that will be the first chain.
3:15
Then we're going to create a second chain.
3:18
In this second chain, we'll take in the company name
3:21
and then output a 20-word description of that company.
3:26
And so you can imagine how these chains might want
3:28
to be run one after another, where the output of the first chain,
3:32
the company name, is then passed into the second chain.
3:35
We can easily do this by creating a simple
3:38
sequential chain where we have the two chains described there.
3:41
And we'll call this overall simple chain. Now,
3:45
what you can do
3:47
is run this chain over any product description.
3:51
And so if we use it with the product above,
3:54
the queen size sheet set, we can run it over
3:56
and we can see that it first outputs royal betting.
3:59
And then it passes it into the second
4:01
chain and it comes up with this description
4:03
of what that company could be about.
4:07
The simple sequential chain works well when there's
4:09
only a single input and a single output.
4:11
But what about when there are multiple inputs or multiple outputs?
4:15
And so we can do this by using just the regular sequential chain.
4:19
So let's import that. And then you're going to create a bunch of
4:22
chains that we're going to use one after another. We're going to
4:25
be using the data from above, which has a review.
4:27
4:28
And so the first chain, we're going to take
4:30
the review and translate it into English.
4:34
With the second chain, we're going to create a summary of
4:39
that review in one sentence.
4:42
And this will use the previously generated English
4:45
review.
4:48
The third chain is going to detect what
4:50
the language of the review was in the first place.
4:53
And so if you notice, this is using the
4:56
review variable that is coming from the original review.
5:01
And finally, the fourth chain will take in multiple inputs.
5:04
So this will take in the summary variable,
5:06
which we calculated with the second chain,
5:09
and the language variable, which we calculated with the third
5:11
chain.
5:12
And it's going to ask for a follow-up response to the summary in
5:16
the specified language.
5:18
One important thing to note about all these subchains
5:21
is that the input keys and output keys
5:23
need to be pretty precise.
5:25
So here, we're taking in review. This is a variable that will be
5:28
passed in at the start.
5:30
We can see that we explicitly set the
5:32
output key to English review. This is then used in the next prompt,
5:36
down below, where we take in English review with that same
5:40
variable name.
5:41
And we set the output key of that chain to summary,
5:44
which we can see is used in the final chain.
5:47
The third prompt takes in review, the original variable,
5:51
and outputs language,
5:52
which is again used in the final prompt.
5:55
It's really important to get these variable names lined
5:58
up exactly right, because there's so many different inputs and
6:00
outputs going on. And if you get any key errors, you should
6:03
definitely check
6:05
that they are lined up so.
6:07
The simple sequential chain takes in multiple chains,
6:09
where each one has a single input and a single output.
6:14
To see a visual representation of this, we can look at the slide,
6:17
where it has one chain
6:19
feeding into the other chain, one after another.
6:23
Here we can see a visual description of the sequential chain.
6:27
Comparing it to the above chain, you can notice that any
6:30
step in the chain can take in multiple input variables.
6:34
This is useful when you have more complicated downstream
6:37
chains that need to be a composition of multiple
6:39
previous chains.
6:42
Now that we have all these chains, we can easily combine
6:45
them in the sequential chain. You'll notice here that we'll pass
6:49
in the four chains we created into the
6:51
chains variable. We'll create the inputs variable with the one
6:55
human input, which is the review.
6:57
And then we want to return all the intermediate outputs.
7:00
So the English review, the summary, and then the follow-up message.
7:06
Now, we can run this over some of the data.
7:09
So let's choose a review and pass it in through the overall chain.
7:19
We can see here that the original review
7:21
looks like it was in French.
7:24
We can see the English review as a translation. We can see a summary
7:28
of that review and then we can see a follow-up
7:31
message in the original language of French.
7:34
You should pause the video here and try
7:36
putting in different inputs.
7:38
So far we've covered the LLM chain and then a sequential chain.
7:42
But what if you want to do something more complicated?
7:45
A pretty common but basic operation is to
7:47
route an input to a chain depending on
7:50
what exactly that input is.
7:52
A good way to imagine this is if you have multiple sub chains,
7:56
each of which specialized for a particular type of input,
8:00
you could have a router chain which first
8:02
decides which subchain to pass it to and then passes it to
8:05
that chain.
8:07
For a concrete example, let's look at where we
8:10
are routing between different types of chains depending
8:13
on the subject that seems to come in. So
8:16
we have here different prompts. One prompt is good for
8:20
answering physics questions. The second prompt is good
8:23
for answering math questions, the third for history, and
8:26
then a fourth for computer science. Let's define
8:29
all these prompt templates.
8:33
After we have these prompt templates, we can then provide
8:36
more information about them.
8:38
We can give each one a name and then a description.
8:41
This description for the physics one is good for
8:44
answering questions about physics. This information is going to
8:47
be passed to the router chain, so the router chain can
8:51
decide when to use this subchain.
8:58
Let's now import the other types of chains that we need.
9:02
Here we need a multi-prompt chain.
9:04
This is a specific type of chain that is used when routing
9:07
between multiple different prompt templates.
9:09
As you can see, all the options we have are prompt templates
9:13
themselves.
9:13
But this is just one type of thing that you can route between.
9:16
You can route between any type of chain.
9:19
The other classes that we'll implement here are an
9:21
LLM router chain.
9:23
This uses a language model itself to route
9:25
between the different subchains. This is where the
9:27
description and the name provided above will be used.
9:31
We'll also import a router output parser.
9:34
This parses the LLM output into a dictionary
9:37
that can be used downstream to determine which
9:40
chain to use and what the input to that chain should be.
9:45
Now we can get around to using it. First, let's import and define the language
9:51
model that we will use.
9:55
We now create the destination chains. These are the chains
9:58
that will be called by the router chain.
10:01
As you can see, each destination chain itself
10:03
is a language model chain, an LLM chain.
10:07
In addition to the destination chains, we also need a default chain.
10:11
This is the chain that's called when the router can't decide
10:15
which of the subchains to use. In the example above, this might
10:19
be called when the input question has nothing
10:22
to do with physics, math, history, or computer science.
10:26
Now we define the template that is used by the LLM to
10:30
route between the different chains.
10:33
This has instructions of the task to be done, as well as the
10:37
specific formatting that the output should be in.
10:40
Let's put a few of these pieces together to
10:42
build the router chain.
10:44
First, we create the full router template by formatting
10:46
it with the destinations that we defined above.
10:49
This template is flexible to a bunch of
10:51
different types of destinations.
10:53
One thing you can do here is pause and add different types
10:56
of destinations.
10:57
So up here, rather than just physics, math, history, and computer science,
11:00
you could add a different subject, like English or Latin.
11:04
Next, we create the prompt template from this template,
11:07
and then we create the router chain by
11:09
passing in the LLM and the overall router prompt.
11:13
Note that here we have the router output parser.
11:17
This is important as it will help this chain
11:20
decide which subchains to route between.
11:24
And finally, putting it all together, we can create
11:27
the overall chain. This has a router chain,
11:30
which is defined here. It has destination chains,
11:33
which we pass in here. And then we also
11:36
pass in the default chain. We can now use this chain.
11:40
So let's ask it some questions.
11:42
If we ask it a question about physics, we should hopefully see
11:46
that it is routed to the physics chain
11:49
with the input, what is blackbody radiation?
11:51
And then that is passed into the chain down below,
11:55
and we can see that the response is very detailed
11:58
with lots of physics details.
12:01
You should pause the video here and try
12:04
putting in different inputs. You can try with all
12:07
other types of special chains that we have defined above.
12:11
So, for example, if we ask it a math question,
12:21
we should see that it's routed to the math chain
12:24
and then passed into that.
12:31
We can also see what happens when we pass in
12:33
a question that is not related to any of the subchains.
12:36
So here, we ask it a question about biology and
12:39
we can see the chain that it chooses is none.
12:42
This means that it will be passed to
12:44
the default chain which itself is just a
12:46
generic call to the language model. The language model luckily
12:48
knows a lot about biology so it can help us out here.
12:52
Now that we've covered these basic building blocks types
12:54
of chains we can start to put them together to create really
12:58
interesting applications. For example, in the next section, we're
13:01
going to cover how to create a chain that can do question answering
13:04
over your documents.

# 第五节

One of the most common, complex applications that
0:05
people are building using an LLM is a system that can answer
0:10
questions on top of or about a document.
0:13
So, given a piece of text, maybe extracted from a
0:17
PDF file or from a webpage or from some company's
0:20
intranet internal document collection, can you use an LLM to answer
0:23
questions about the content of those documents to help
0:26
users gain a deeper understanding and
0:28
get access to the information that they need?
0:31
This is really powerful because it starts to combine
0:34
these language models with data that they weren't
0:36
originally trained on. So it makes them much
0:39
more flexible and adaptable to your use case. It's also
0:42
really exciting because we'll start to move beyond language models, prompts, and output
0:46
parsers and start introducing some more of the key components
0:49
of LangChain, such as embedding models and vector stores.
0:52
0:53
As Andrew mentioned, this is one of the
0:54
more popular chains that we've got, so I hope you're excited.
0:56
In fact, embeddings and vector stores
0:58
are some of the most powerful modern techniques,
1:01
so if you have not seen them yet, they are very much worth learning
1:06
about. So with that, let's dive in!
1:08
Let's do it! So we're going to start by importing
1:11
the environment variables as we always do.
1:14
Now we're going to import some things that will help
1:17
us when building this chain. We're going to import the retrieval
1:20
QA chain. This will do retrieval over some documents. We're going to
1:23
import our favorite chat open AI language model. We're going
1:25
to import a document loader. This is going to
1:27
be used to load some proprietary data that we're going to combine with
1:31
the language model. In this case it's going
1:33
to be in a CSV.
1:34
So we're going to import the CSV loader. Finally
1:37
we're going to import a vector store.
1:39
There are many different types of vector stores and we'll
1:42
cover what exactly these are later on but we're going
1:44
to get started with the "DocArrayInMemorySearch" vector store.
1:47
This is really nice because it's an in-memory
1:49
vector store and it doesn't require connecting to an
1:51
external database of any kind so it makes
1:53
it really easy to get started.
1:56
We're also going to import display and markdown to common
1:59
utilities for displaying information in Jupyter
2:01
notebooks.
2:02
We've provided a CSV of outdoor clothing that we're going
2:06
to use to combine with the language model. Here we're
2:09
going to initialize a loader, the CSV loader,
2:12
with a path to this file.
2:15
We're next going to import an index, the "VectorStoreIndexCreator". This
2:20
will help us create a vector store really easily.
2:24
As we can see below, there will only be a few
2:27
lines of code to create this.
2:32
To create it, we're going to specify two things.
2:35
First, we're going to specify the vector store class.
2:39
As mentioned before, we're going to use this vector store, as
2:41
it's a particularly easy one to get started with.
2:44
After it's been created, we're then going to call "from_loaders", which takes in
2:48
a list of document loaders.
2:50
We've only got one loader that we really care about, so that's what
2:53
we're passing in here.
2:56
It's now been created and we can start to
2:58
ask questions about it.
3:00
Below we'll cover what exactly happened under the hood, so let's not
3:03
worry about that for now. Here, we'll start with a query.
3:07
We'll then create a response using "index.query" and pass in this query.
3:13
3:15
Again, we'll cover what's going on under the hood down below.
3:18
3:19
For now, we'll just wait for it to respond.
3:27
After it finishes, we can now take a look at what exactly was returned.
3:33
We've gotten back a table in markdown with names
3:36
and descriptions for all shirts with sun protection.
3:39
We've also got a nice little summary that the
3:41
language model has provided us.
3:44
So we've gone over how to do question answering over your
3:47
documents, but what exactly is going on underneath the hood? First, let's
3:50
think about the general idea. We want to use language models and
3:54
combine it with a lot of our documents.
3:57
But there's a key issue.
3:58
Language models can only inspect a few thousand
4:00
words at a time.
4:01
So if we have really large documents, how can we get
4:04
the language model to answer questions about everything
4:06
that's in there?
4:08
This is where embeddings and vector stores come into play. First,
4:13
let's talk about embeddings.
4:15
Embeddings create numerical representations
4:17
for pieces of text.
4:20
This numerical representation captures the semantic
4:22
meaning of the piece of text that it's been run over.
4:26
Pieces of text with similar content will have similar vectors.
4:30
This lets us compare pieces of text in the vector space.
4:33
In the example below, we can see that
4:35
we have three sentences.
4:37
The first two are about pets, while the third is about a car.
4:41
If we look at the representation in the numeric space,
4:44
we can see that when we compare the two vectors on the
4:48
pieces of text corresponding to the sentences about pets, they're
4:51
very similar.
4:52
While if we compare it to the one that talks about a car,
4:54
they're not similar at all.
4:56
This will let us easily figure out which pieces of
4:59
text are like each other, which will be very useful as
5:02
we think about which pieces of text we want to include when
5:05
passing to the language model to answer a question.
5:08
The next component that we're going to cover is
5:10
the vector database.
5:11
A vector database is a way to store these
5:14
vector representations that we created in the previous step.
5:16
The way that we create this vector database
5:18
is we populate it with chunks of text
5:21
coming from incoming documents.
5:22
When we get a big incoming document, we're first going to break it
5:25
up into smaller chunks.
5:27
This helps create pieces of text that are
5:29
smaller than the original document, which is useful because
5:31
we may not be able to pass the whole document to the
5:35
language model. So we want to create these small chunks
5:37
so we can only pass the most relevant
5:39
ones to the language model.
5:41
We then create an embedding for each of these chunks,
5:44
and then we store those in a vector database. That's
5:47
what happens when we create the index.
5:50
Now that we've got this index, we can use it during
5:52
runtime to find the pieces of text most
5:55
relevant to an incoming query.
5:56
When a query comes in, we first create an
5:59
embedding for that query.
6:00
We then compare it to all the vectors
6:02
in the vector database, and we pick the n most similar.
6:06
These are then returned, and we can pass those in the prompt
6:09
to the language model to get back a final answer. So above,
6:12
we created this chain and only a few lines of code.
6:16
That's great for getting started quickly.
6:18
But let's now do it a bit more step-by-step and understand what exactly is going on
6:22
under the hood. The first step is similar to above.
6:25
We're going to create a document loader, loading
6:28
from that CSV with all the descriptions of the
6:31
products that we want to do question answering over.
6:34
We can then load documents from this document loader.
6:38
6:39
If we look at the individual documents, we can see that each
6:44
document corresponds to one of the products in the CSV.
6:48
Previously, we talked about creating chunks.
6:51
Because these documents are already so small,
6:52
we actually don't need to do any chunking here.
6:55
And so we can create embeddings directly.
6:59
To create embeddings, we're going to use OpenAI's embedding
7:02
class.
7:03
We can import it and initialize it here.
7:06
If we want to see what these embeddings do,
7:09
we can actually take a look at what happens when we embed
7:12
a particular piece of text.
7:19
Let's use the "embed_query" method on the embeddings object to create an
7:22
embeddings for a particular piece of text. In this
7:25
case, the sentence, "Hi, my name is Harrison."
7:30
If we take a look at this embedding, we can see that there are
7:33
over a thousand different elements.
7:39
Each of these elements is a different numerical value.
7:43
Combined, this creates the overall numerical representation
7:46
for this piece of text.
7:49
We want to create embeddings for all the
7:51
pieces of text that we just loaddand then
7:53
we also want to store them in a vector store.
7:57
We can do that by using the "from_documents" method
8:00
on the vector store.
8:02
This method takes in a list of documents,
8:05
an embedding object, and then we'll create an overall vector store.
8:10
We can now use this vector store to find pieces of text
8:13
similar to an incoming query.
8:16
So let's look at the query, "Please suggest a shirt with sunblocking".
8:20
8:21
If we use the similarity search method on the vector
8:23
store and pass in a query, we will get back a list of documents.
8:27
8:35
We can see that it returns four documents,
8:38
and if we look at the first one, we can see that it
8:43
is indeed a shirt about sunblocking.
8:46
So, how do we use this to do question
8:49
answering over our own documents? First, we need to create a
8:52
retriever from this vector store. A retriever is a
8:55
generic interface that can be underpinned by any
8:58
method that takes in a query and returns documents.
9:01
Vector stores and embeddings are one such method to do so,
9:04
although there are plenty of different methods,
9:07
some less advanced, some more advanced.
9:10
Next, because we want to do text generation and
9:12
return a natural language response, we're going to import a
9:15
language model and we're going to use ChatOpenAI.
9:19
If we were doing this by hand, what we would do is
9:22
we would combine the documents into a single piece of text.
9:27
So we'd do something like this, where we join all the
9:31
page content in the documents into a variable
9:34
and then would pass this variable or a
9:36
variant on the question, like, "Please list all your shirts
9:40
with sun protection in a table in markdown
9:43
and summarize each one." into the language model.
9:47
And if we print out the response here, we can see that we get back a
9:51
table exactly as we asked for.
9:54
All of those steps can be encapsulated with the
9:57
LangChain chain.
9:58
So here we can create a retrieval QA chain.
10:01
This does retrieval and then does question answering over
10:03
the retrieved documents. To create such a chain, we'll
10:06
pass in a few different things. First, we'll pass
10:08
in the language model.
10:10
This will be used for doing the text generation at the end.
10:14
Next, we'll pass in the chain type. We're going to use "stuff". This is
10:17
the simplest method as it just stuffs all
10:20
the documents into context and makes one call
10:22
to a language model.
10:23
There are a few other methods that you can use
10:25
to do question answering that I'll maybe touch on at the end, but
10:28
we're not going to look at in detail.
10:30
Third, we're going to pass in a retriever.
10:32
The retriever we created above is just an
10:35
interface for fetching documents.
10:37
This will be used to fetch the documents and pass
10:38
it to the language model.
10:40
And then finally, we're going to set "verbose=True".
10:44
Now, we can create a query and we can run the chain on
10:49
this query.
11:06
When we get the response, we can again display it using
11:10
the display and markdown utilities. You can pause the video
11:13
here and try it out with a bunch of different queries.
11:19
So that's how you do it in detail, but remember that we can still
11:21
do it pretty easily with just the one line that
11:23
we had up above.
11:25
So, these two things equate to the same result.
11:28
And that's part of the interesting stuff about LangChain. You
11:30
can do it in one line, or you can look at
11:33
the individual things and break it down into
11:35
five more detailed ones.
11:36
The five more detailed ones let you set
11:38
more specifics about what exactly is going on, but the one-liner
11:41
is easy to get started. So up to you as to how you'd prefer
11:45
to go forward.
11:46
We can also customize the index when we're creating it. And
11:49
so if you remember, when we created it by hand, we
11:52
specified an embedding. And we can specify an
11:55
embedding here as well. And so this will give us flexibility
11:58
over how the embeddings themselves are
12:00
created. And we can also swap out the vector store
12:02
here for a different type of vector store. So
12:05
there's the same level of customization that you did when you create
12:09
it by hand that's also available when you create the
12:11
index here.
12:13
We use the "stuff method" in this notebook.
12:15
The stuff method is really nice because it's
12:17
pretty simple. You just put all of it into one prompt and send that to
12:21
the language model and get back one response.
12:23
So it's quite simple to understand what's going
12:25
on. It's quite cheap and it works pretty well.
12:28
But that doesn't always work okay.
12:31
So if you remember, when we fetched the
12:33
documents in the notebook, we only got four documents back
12:35
and they were relatively small.
12:37
But what if you wanted to do the same type of question
12:40
answering over lots of different types of chunks?
12:43
Then there are a few different methods that we can use.
12:45
The first is "Map_reduce".
12:47
This basically takes all the chunks, passes them along with the
12:50
question to a language model, gets back a response, and then uses
12:54
another language model call to summarize all of the
12:57
individual responses into a final answer.
13:00
This is really powerful because it can operate
13:03
over any number of documents.
13:05
And it's also really powerful because you can do the
13:08
individual questions in parallel.
13:10
But it does take a lot more calls. And it does treat
13:13
all the documents as independent, which may not always
13:16
be the most desired thing. "Refine", which is another method,
13:19
is again used to loop over many documents.
13:22
But it actually does it iteratively. It builds upon the
13:25
answer from the previous document.
13:27
So this is really good for combining information and
13:30
building up an answer over time. It will generally lead to longer
13:34
answers.
13:35
And it's also not as fast because now the calls aren't independent.
13:38
They depend on the result of previous calls.
13:41
This means that it often takes a good
13:44
while longer and takes just as many calls as "Map_reduce", basically.
13:47
"Map_rerank" is a pretty interesting and a bit more
13:50
experimental one where you do a single call to the language model
13:54
for each document. And you also ask it to return a score.
13:58
13:58
And then you select the highest score.
14:01
This relies on the language model to know
14:03
what the score should be. So you often have to tell it, "Hey,
14:06
it should be a high score if it's relevant to the document and really
14:10
refine the instructions there". Similar to "Map_reduce", all
14:12
the calls are independent. So you
14:14
can batch them and it's relatively fast. But again, you're making a bunch
14:17
of language model calls. So it will be
14:19
a bit more expensive.
14:21
The most common of these methods is the "stuff method",
14:23
which we used in the notebook to combine
14:25
it all into one document.
14:27
The second most common is the "Map_reduce" method, which takes these chunks
14:31
and sends them to the language model.
14:34
These methods here, stuff, map_reduce, refine, and rerank can also
14:37
be used for lots of other chains besides just
14:39
question answering.
14:41
For example, a really common use case of the "Map_reduce"
14:44
chain is for summarization, where you have a really long document
14:47
and you want to recursively summarize
14:49
pieces of information in it.
14:52
That's it for question answering over documents.
14:54
As you may have noticed, there's a lot going on in the
14:57
different chains that we have here. And so in the next section, we'll cover
15:01
ways to better understand what exactly is going
15:03
on inside all of these chains.

# 第六节

When building a complex application using an LLM, one of the
0:07
important but sometimes tricky steps is how do you
0:10
evaluate how well your application is doing?
0:12
Is it meeting some accuracy criteria?
0:15
And also, if you decide to change your implementation,
0:18
maybe swap in a different LLM, or change the strategy
0:21
of how you use a vector database or something else to retrieve chunks,
0:26
or change some other parameters of your system,
0:28
how do you know if you're making it better or worse? In
0:32
this video, Harrison will dive into some
0:35
frameworks on how to think about evaluating a LLM-based application,
0:38
as well as some tools to help
0:40
you do that.
0:41
These applications are really chains and sequences of
0:43
a lot of different steps. And so honestly, part of the first
0:47
thing that you should do is just understand
0:49
what exactly is going in and coming out of each step.
0:52
And so some of the tools can really just be thought of
0:55
as visualizers or debuggers in that vein.
0:58
But it's often really useful to get a more holistic picture on a lot
1:02
of different data points of how the model is doing. And
1:04
one way to do that is by looking at things by eye. But
1:08
there's also this really cool idea of using language models themselves
1:10
and chains themselves to evaluate other
1:12
language models, and other chains, and other
1:14
applications. And we'll dive a bunch into that as well.
1:16
So, lots of cool topics, and I find that with a lot of
1:22
development shifting towards prompting-based development, developing
1:24
applications using LLMs, this whole workflow evaluation process
1:28
is being rethought. So, lots of exciting concepts in this video.
1:32
Let's dive in.
1:34
Alright, so let's get set up with evaluation. First, we
1:38
need to have the chain or the application that we're going
1:41
to evaluate in the first place.
1:44
And we're going to use the document question answering chain
1:47
from the previous lesson.
1:49
So we're going to import everything we need.
1:51
We're going to load the same data that we were using.
1:57
We're going to create that index with one line.
2:01
And then we're going to create the retrieval QA chain by
2:06
specifying the language model, the chain type, the retriever, and
2:10
then the verbosity that we're going to print out.
2:15
So we've got this application, and the first thing we need to do is
2:19
we need to really figure out what are
2:22
some data points that we want to evaluate it on.
2:25
And so there's a few different methods that we're going
2:27
to cover for doing this.
2:29
The first is the most simple, which is basically we're
2:32
going to come up with data points that we think
2:35
are good examples ourselves.
2:37
And so to do that, we can just look at some of the data and come up
2:42
with example questions and then example ground truth
2:44
answers that we can later use to evaluate.
2:48
So if we look at a few of the documents here,
2:51
we can kind of get a sense of what's going on inside them. It
2:56
looks like the first one, there's this pullover set, there's this in
3:00
the second one, there's this jacket, it has a bunch of details about all of
3:05
them.
3:06
And from these details, we can create some
3:09
example query and answer pairs.
3:11
So the first one, we can ask a simple,
3:14
"Does the Cozy Comfort Pullover Set have side pockets?".
3:17
And we can see by looking above that it does, in fact,
3:21
have some side pockets in it.
3:24
And then for the second one, we can see that this
3:27
jacket is from a certain collection, the DownTek collection.
3:29
And so we can ask the question, "What collection is this jacket from?"
3:33
3:34
And have the answer be, "The DownTek collection". So here we've
3:37
created two examples.
3:39
But this doesn't really scale that well. It
3:41
takes a bit of time to look through each example
3:43
and figure out what's going on. And so is there a way that
3:47
we can automate it?
3:48
And one of the really cool ways that
3:50
we think we can automate it is with language models themselves.
3:54
So we have a chain in LangChain that can do exactly that.
3:58
So we can import the QA generation chain.
4:00
And this will take in documents and it
4:03
will create a question answer pair from each document.
4:06
It'll do this using a language model itself. So
4:09
we need to create this chain by passing in the
4:12
Chat OpenAI language model.
4:14
And then from there, we can create a bunch of examples.
4:17
And so we're going to use the apply and parse method
4:21
because this is applying an output parser to
4:23
the result because we want to get back
4:25
a dictionary that has the query and answer pair, not
4:28
just a single string.
4:36
And so now, if we look at what exactly is returned here,
4:39
we can see a query, and we can see an answer, and let's
4:43
check the document that this is a question and answer for, and
4:46
we can see that it's asking what the weight of this is, we can
4:50
see that it's taking the weight from here, and look at
4:53
that! We just generated a bunch of question-answer pairs. We didn't
4:56
have to write it all ourselves. Saves us a bunch of
5:00
time, and we can do more exciting things. And so now, let's go
5:03
ahead and add these examples into the examples that we already created.
5:07
5:08
So, we got these examples now, but how exactly do we evaluate what's
5:12
going on?
5:13
The first thing we want to do is just run
5:16
an example through the chain, and take a look at the output it
5:19
produces.
5:20
So here we pass in a query, and we get back an answer. But,
5:24
this is a little bit limiting in terms
5:26
of what we can see that's actually happening inside the chain.
5:30
5:30
What is the actual prompt that's going into the language model? What
5:34
are the documents that it retrieves? If this were a more
5:37
complex chain with multiple steps in it, what
5:39
are the intermediate results?
5:41
It's oftentimes not enough to just look at the final answer to understand
5:45
what is or could be going wrong in the chain.
5:49
And to help with that, we have a fun little util in LangChain called
5:55
"langchain.debug".
5:58
And so if we set "langchain.debug = True", and we now rerun
6:03
the same example as above.
6:06
We can see that it starts printing out
6:08
a lot more information.
6:11
And so if we look at what exactly it's printing out, we can see that it's diving
6:15
down first into the retrieval QA chain.
6:17
And then it's going down into a stuff documents chain. And
6:20
so as mentioned, we're using the stuff method.
6:23
And now it's entering the LLM chain, where we have a few different inputs. So
6:27
we can see the original question is right there.
6:31
And now we're passing in this context. And
6:33
we can see that this context is created from a bunch of
6:36
the different documents that we've retrieved.
6:39
And so when doing question answering, oftentimes when a wrong result is
6:43
returned, it's not necessarily the language model itself that's messing
6:46
up. It's actually the retrieval step that's messing up.
6:49
6:50
And so taking a really close look at
6:52
what exactly the question is, and what exactly the context is,
6:55
can help debug what's going wrong.
6:58
We can then step down one more level
7:01
and see exactly what is entering the language model,
7:04
Chat OpenAI itself.
7:05
And so here we can see the full prompt that's
7:08
passed in. So we've got a system message.
7:10
We've got the description of the prompt that's used. And so
7:13
this is the prompt that the question answering chain is using under
7:16
the hood, which we actually haven't even looked
7:18
at until now.
7:19
And so we can see the prompt printing out,
7:22
"Use the following pieces of context to answer the user's
7:25
question. If you don't know the answer, just say
7:27
that you don't know, don't try to make up an answer." And then we see
7:31
a bunch of the context as inserted before, and then we see
7:35
a human question, which is the question that we asked it. We can also see
7:39
a lot more information about the actual return
7:41
type. So rather than just a string, we get back a
7:44
bunch of information like the "token_usage", so the "prompt_tokens", the
7:47
"completion_tokens", "total_tokens", and the "model_name".
7:48
7:49
And this can be really useful to track the tokens that you're
7:53
using in your chains or calls to language models over time and
7:56
keep track of the total number of tokens, which
7:59
corresponds very closely to the total cost.
8:02
And because this is a relatively simple chain,
8:04
we can now see that the final response, "The Cozy Comfort Pullover Set,
8:08
Stripe does have side pockets.", is getting bubbled up
8:11
through the chains and getting returned to the user. So we've
8:14
just walked through how to look at and debug what's going
8:17
on with a single input to this chain.
8:21
But what about all the examples we created?
8:23
How are we going to evaluate those?
8:25
Similarly to when creating them, one way to do it would be manually.
8:29
We could run the chain over all the examples, then look at the outputs,
8:33
and try to figure out what's going on, whether it's correct, incorrect, partially
8:37
correct. Similar to creating the examples, that starts
8:39
to get a little bit tedious over time.
8:42
And so let's go back to our favorite solution. Can
8:45
we ask a language model to do it? First, we need to create predictions
8:50
for all the examples. Before doing that, I'm actually
8:53
going to turn off the debug mode in order to just not
8:57
print everything out onto the screen.
9:00
And then I'm going to create predictions for all
9:03
the different examples. And so I think we
9:05
had seven examples total, and so we're going to loop through this
9:09
chain seven times, getting a prediction for each one.
9:28
Now that we've got these examples, we can think about evaluating them.
9:33
So we're going to import the QA, question answering, eval chain. We
9:38
are going to create this chain with a language model, because
9:42
again, we're going to be using a language model to help
9:46
do the evaluation.
9:50
And then we're going to call evaluate on this chain. We're
9:53
going to pass in examples and predictions, and we're going to
9:57
get back a bunch of graded outputs. And so, in order to
10:01
see what exactly is going on for each example, we're going to loop
10:05
through them.
10:07
We're going to print out the question, and
10:09
again, this was generated by a language model.
10:11
We're going to print out the real answer, and
10:13
again, this was also generated by a language model when it had the whole
10:17
document in front of it, and so it could generate
10:19
a ground truth answer.
10:21
We're going to print out the predicted answer, and
10:23
this is generated by a language model when it's doing
10:26
the QA chain, when it's doing the retrieval with the embeddings and the
10:29
vector databases, passing that into a language model, and then trying
10:32
to guess the predicted answer.
10:34
And then we're also going to print out the grade, and
10:37
again, this is also generated by a
10:39
language model when it's asking the eval chain to grade what's going on
10:42
and whether it's correct or incorrect.
10:44
And so, when we loop through all these examples and print them out,
10:48
we can see those in detail for each example.
10:52
And looks like here it got everything correct. This is a
10:56
relatively simple retrieval problem, so that is reassuring.
11:00
So let's look at the first example. The question here is, "Does
11:04
the Cozy Comfort Pullover Set have side pockets?". The
11:07
real answer, and we created this, is "Yes".
11:10
The predicted answer, which the language model produced was,
11:14
"The Cozy Comfort Pullover Set, Stripe does have side pockets".
11:19
And so we can understand that this is a correct answer.
11:22
And actually the language model does as well,
11:24
and it grades it correct.
11:26
But let's think about why we actually need to use the language
11:28
model in the first place.
11:30
These two strings are actually nothing alike.
11:35
They're very different. One's really short, one's really long. I don't even
11:39
think, yes doesn't appear anywhere in this string. So if we were to
11:43
try to do some string matching, or exact matching, or even
11:47
some regexes here, it wouldn't know what to
11:49
do. They're not the same thing. And that shows off the importance of
11:54
using the language model to do evaluation here. You've got these answers,
11:58
which are arbitrary strings. There's no single one
12:00
truth string that is the best possible answer. There's
12:03
many different variants. And as long as they have
12:06
the same semantic meaning, they should be graded as
12:09
being similar. And that's what a language model helps with, as opposed
12:13
to just doing exact matching.
12:15
12:16
This difficulty in comparing strings is
12:18
what makes evaluation of language models so hard in
12:21
the first place.
12:23
We're using them for these really open-ended tasks,
12:25
where they're asked to generate text. This hasn't really
12:28
been done before, as models until recently
12:30
weren't really good enough to do this. And so a lot of
12:33
the evaluation metrics that did exist up to
12:36
this point just aren't good enough. And we're having
12:38
to invent new ones, and invent new heuristics for doing so.
12:42
And the most interesting and most popular of those heuristics at the moment
12:45
is actually using a language model to do the evaluation. This finishes
12:49
the evaluation lesson, but one last thing I want to
12:52
show you is the LangChain Evaluation Platform. This is a way to
12:55
do everything that we just did in the notebook, but
12:58
persisted and show it in a UI. And so let's check it out. Here,
13:02
we can see that we have a session. We called it, deeplearningai.
13:06
13:06
And we can see here that we've actually persisted all the
13:09
runs that we ran in the notebook. And
13:12
so this is a good way to track the inputs
13:14
and outputs at a high level.
13:16
But it's also a really good way to see what
13:19
exactly is going on underneath. So this is the same
13:22
information that was printed out in the notebook
13:24
when we turned on debug mode.
13:26
But it's just visualized in a UI in a little bit of a nicer way.
13:31
And so we can see the inputs to the chain and the
13:33
outputs to the chain at each step. And then we can
13:36
click further and further down into the chain
13:38
and see more and more information about what
13:40
is actually getting passed in. And so if we go all
13:42
the way down to the bottom, we can now see what's
13:45
getting passed exactly to the chat model. We've got
13:47
the system message here.
13:49
We've got the human question here. We've got the response from the
13:52
chat model here. And we've got some output metadata. One other thing
13:56
that we've added here is the ability to add these examples to a
14:00
data set. So if you remember, when creating those data
14:03
sets of examples at the start, we created them partially
14:06
by hand, partially with a language model.
14:09
Here, we can add it to a data set by clicking on this little button.
14:14
And we now have the input query and the output results.
14:17
And so we can create a data set. We can call it deep learning.
14:23
And then we can start adding examples to this data set.
14:26
And so again, getting back to the original thing that we tackled
14:29
at the beginning of the lesson, we need to create
14:32
these data sets so that we can do evaluation. This is a really
14:35
good way to have this just running in the background.
14:38
And then add to the example data sets over time and start
14:42
building up these examples that you can start using for evaluation and
14:46
have this flywheel of evaluation start turning.

# 第七节

Sometimes people think of a large language model
0:05
as a knowledge store, as if it's learned to memorize a
0:08
lot of information, maybe off the internet, so
0:11
when you ask it a question, it can answer the question.
0:15
But I think a even more useful way to think of a
0:18
large language model is sometimes as a reasoning engine,
0:21
in which you can give it chunks of text or other sources
0:24
of information.
0:25
And then the large language model, LLM, will maybe use
0:29
this background knowledge that's learned off the internet, but
0:32
to use the new information you give it
0:34
to help you answer questions or reason through
0:37
content or decide even what to do next. And that's what
0:41
LangChain's Agents framework helps you to do.
0:43
0:43
Agents are probably my favorite part of LangChain.
0:45
I think they're also one of the most powerful parts, but
0:49
they're also one of the newer parts. So we're
0:51
seeing a lot of stuff emerge here that's really new to
0:54
everyone in the field. And so this should be a very exciting lesson
0:58
as we dive into what agents are, how to create and
1:01
how to use agents, how to equip them with different types of
1:04
tools, like search engines that come built into LangChain,
1:07
and then also how to create your own
1:09
tools, so that you can let agents interact with any
1:12
data stores, any APIs, any functions that you
1:14
might want them to.
1:15
So this is exciting, cutting-edge stuff, but
1:18
already with emerging, important use cases. So with that, let's
1:23
dive in!
1:23
So to get started with Agents, we're going to start as we always do, by
1:29
importing the correct environment variables.
1:31
We're also going to need to install a few packages here. So
1:34
we're going to use the "duckduckgo-search" engine and "wikipedia". So we're
1:37
going to want to pip install those. I've already installed those
1:40
in my environment, so I'm not going to run this
1:43
line. But if you guys have not, you should uncomment that line, run
1:47
it, and then you're good to go.
1:49
1:50
We're then going to import some methods and classes
1:52
that we need from LangChain. So we're going to import some methods
1:56
to load tools.
1:57
And these are things that we're going to connect the
2:00
language model to. We're going to load a method to
2:03
initialize the agent. We're going to load the Chat OpenAI
2:05
wrapper, and we're going to load "AgentType".
2:08
So "AgentType" will be used to specify what
2:10
type of agent we want to use. There are a bunch of different
2:14
types of agents in LangChain. We're not going to go over
2:17
all of them right now. We'll just choose one and run with that. We're then
2:21
going to initialize the language model that we're going to use.
2:24
Again, we're using this as the reasoning engine that we're
2:26
using to drive the agent.
2:29
We'll then load the tools that we're going to use.
2:32
So we're going to load DuckDuckGo search and Wikipedia. And
2:36
these are built-in LangChain tools. Finally, we're going to
2:40
initialize the agent.
2:41
We pass it the tools, the language model, and agent type.
2:45
2:46
And so here we're using "CHAT_ZERO_SHOT_REACT_DESCRIPTION".
2:50
Not going to go over in too much detail what this means.
2:53
The important things to note are chat. This is optimized to
2:56
work with chat models.
2:57
And then react. React is a prompting strategy that
3:01
elicits better thoughts from a language model.
3:04
We're also going to set "handle_parsing_errors=True".
3:08
If you remember from the first lesson, we chatted a bunch
3:11
about output parsers and how those can be used to take the
3:14
LLM output, which is a string, and parse it into
3:17
a specific format that we can use downstream.
3:20
That's extremely important here. When we take the output of the LLM, which
3:24
is text, and parse it into the specific action and the specific
3:28
action input that the language model should take.
3:31
Let's now use this agent. Let's ask it a question about a recent
3:35
event that the model, when it was trained, didn't know. So let's ask
3:40
it about the 2022 World Cup.
3:42
The models here were trained on data up to around 2021.
3:46
So it shouldn't know the answer to this question. And
3:49
so it should realize that it needs to use a
3:52
tool to look up this recent piece of information.
4:00
So we can see here that the agent realizes that it needs
4:06
to use DuckDuckGo search, and then looks up
4:10
the 2022 World Cup winner.
4:13
And so it gets back a bunch of information. We can then see
4:17
that the agent thinks that the 2022 World
4:20
Cup has not happened yet.
4:22
So this is a good example of why agents are still pretty exploratory.
4:26
You can see that there's a bunch of information here
4:29
about the 2022 World Cup, but it doesn't quite
4:32
realize all that has happened. And so it needs to look up more things and
4:37
get more information. And so then based on that information, it can
4:40
respond with the correct answer that "Argentina won
4:43
the 2022 World Cup".
4:46
Let's then ask it a question where it should
4:48
recognize that it needs to use Wikipedia.
4:51
So Wikipedia, it has a lot of information about specific
4:54
people and specific entities from a long time ago. It doesn't
4:58
need to be current information.
5:01
So let's ask it about Tom M. Mitchell, an
5:04
American computer scientist. And what book did he write? We
5:06
can see here that it recognizes that it
5:09
should use Wikipedia to look up the answer. So
5:11
it searches for Tom M. Mitchell Wikipedia.
5:14
And then does another follow-up search just to make sure that it's got
5:17
the right answer. So it searches for "Tom M. Mitchell Machine
5:20
Learning" and gets back more information.
5:23
And then based on that, it's able to finally answer with "Tom M. Mitchell
5:26
wrote the textbook, 'Machine Learning'".
5:28
You should pause the video here and try
5:30
putting in different inputs.
5:32
So far, we've used tools that come defined in LangChain already, but
5:36
a big power of agents is that you can connect it to
5:39
your own sources of information, your own APIs, your own data.
5:43
So here we're going to go over how you can create
5:45
a custom tool so that you can connect
5:47
it to whatever you want.
5:49
Let's make a tool that's going to tell us what the
5:51
current data is.
5:53
First, we're going to import this tool decorator.
5:55
5:56
This can be applied to any function and
5:58
it turns it into a tool that LangChain can use. Next,
6:01
we're going to write a function called "time",
6:04
which takes in any text string. We're not really going to use that. And it's
6:08
going to return today's date by calling date time.
6:14
In addition to the name of the function, we're also going to write a
6:17
really detailed doc string.
6:19
That's because this is what the agent will use
6:21
to know when it should call this tool and how it should
6:24
call this tool.
6:26
For example, here we say that, "The input should always
6:29
be an empty string.". That's because we don't use it.
6:32
If we have more stringent requirements on what
6:35
the input should be, for example, if we have a
6:38
function that should always take in a search query or a SQL
6:42
statement, you'll want to make sure to mention that here. We're now
6:46
going to create another agent. This time we're adding the
6:49
time tool to the list of existing tools.
6:54
And finally, let's call the agent and ask it
6:56
what the date today is.
7:02
It recognizes that it needs to use the time tool,
7:05
which it specifies here.
7:06
It has the action input as an empty string. This is great.
7:09
This is what we told it to do.
7:11
And then it returns with an observation. And then finally,
7:14
the language model takes that observation
7:17
and responds to the user, "Today's date is 2023-05-21".
7:21
You should pause the video here and try
7:23
putting in different inputs. This wraps up the lesson on agents.
7:27
This is one of the newer and more exciting
7:30
and more experimental pieces of LangChain.
7:33
So I hope you enjoy using it. Hopefully it showed you how you
7:37
can use a language model as a reasoning engine
7:39
to take different actions and connect
7:41
to other functions and data sources.
