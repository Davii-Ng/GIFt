"""Ranking test cases: query | positive paraphrase | two hard negatives.
Labels are heuristic: inspect ambiguous examples before drawing strong conclusions.
"""
RAW = r'''
Subject-Object Reversal|A dog is chasing a cat|A canine is pursuing a cat|A cat is chasing a dog|A dog and a cat are sleeping
Subject-Object Reversal|A man is pushing a woman|A person pushes a woman|A woman is pushing a man|A man is standing beside a woman
Subject-Object Reversal|A girl is following a boy|A girl walks behind a boy following him|A boy is following a girl|A girl and a boy are talking
Subject-Object Reversal|A teacher is helping a student|A teacher gives assistance to a student|A student is helping a teacher|A teacher watches a student
Subject-Object Reversal|A cat is scratching a dog|A feline scratches a dog|A dog is scratching a cat|A cat is sitting next to a dog
Subject-Object Reversal|A boy is hugging a girl|A boy embraces a girl|A girl is hugging a boy|A boy waves at a girl
Subject-Object Reversal|A woman is photographing a man|A woman takes a picture of a man|A man is photographing a woman|A woman speaks to a man
Subject-Object Reversal|A dog is pulling a child|A dog drags a child along|A child is pulling a dog|A dog plays with a child
Negation|A man is not smiling|A man has no smile on his face|A man is smiling|A man is laughing loudly
Negation|A dog is not running|A dog remains still instead of running|A dog is running|A dog runs quickly
Negation|A woman is not dancing|A woman does not dance|A woman is dancing|A woman dances with friends
Negation|The door is not open|The door is closed|The door is open|The door is opening
Negation|A person is not asleep|A person is awake|A person is asleep|A sleeping person lies down
Negation|A child is not crying|A child has stopped crying|A child is crying|A child weeps loudly
Negation|The man is not unhappy|The man is not feeling sad|The man is unhappy|The man is feeling miserable
Negation|A cat is not inside the box|A cat is outside the box|A cat is inside the box|A cat is hiding in a box
Temporal Order|A man sits down and then stands up|A man takes a seat before getting up|A man stands up and then sits down|A man remains seated
Temporal Order|A woman opens a door before entering|A woman opens the door and then walks in|A woman enters before opening a door|A woman closes the door and leaves
Temporal Order|A dog jumps before running away|A dog leaps and then runs off|A dog runs off and then jumps|A dog lies down without moving
Temporal Order|A boy waves and then walks away|A boy waves goodbye before leaving|A boy walks away and then waves|A boy stays and talks
Temporal Order|A person falls before getting up|Someone falls down and afterward stands|Someone stands and afterward falls|Someone stands without falling
Temporal Order|A woman picks up a bag before leaving|A woman grabs her bag and then exits|A woman leaves and then picks up a bag|A woman drops her bag and waits
Temporal Order|A cat lands and then runs|A cat touches the ground before running|A cat runs before landing|A cat remains in the air
Temporal Order|A man drinks water before speaking|A man takes a sip, then starts talking|A man speaks before drinking water|A man speaks without drinking
Spatial Relations|A cat is under the table|A feline sits beneath a table|A cat is on top of the table|A cat is next to a table
Spatial Relations|A dog is inside the house|A dog is indoors in a house|A dog is outside the house|A dog is standing near a house
Spatial Relations|A person stands to the left of a car|Someone is on the car's left side|Someone is on the right side of a car|Someone is sitting inside the car
Spatial Relations|A bird flies above a tree|A bird is flying over a tree|A bird flies below a tree|A bird is perched on a tree
Spatial Relations|A ball rolls behind a couch|A ball moves to the back of a couch|A ball rolls in front of a couch|A ball sits on the couch
Spatial Relations|A child hides behind a curtain|A kid is concealed at the back of a curtain|A child stands in front of a curtain|A child opens the curtain
Spatial Relations|A toy is in the box|A toy is contained inside a box|A toy is outside the box|A toy is on top of the box
Spatial Relations|A cat jumps over a fence|A cat leaps across the top of a fence|A cat crawls under a fence|A cat waits beside the fence
Counting|Exactly one dog is running|A single dog runs|Three dogs are running|Two dogs are running
Counting|Two people are dancing|A pair of people dances|One person is dancing|Five people are dancing
Counting|Three cats are sleeping|A trio of cats is asleep|One cat is sleeping|Two cats are sleeping
Counting|One bird is flying|A single bird is in flight|Four birds are flying|Two birds are flying
Counting|Four children are jumping|Four kids are leaping|One child is jumping|Three children are jumping
Counting|Two cars are moving|A pair of cars is driving|One car is moving|Three cars are moving
Counting|Five dogs are sitting|Five dogs are seated|Two dogs are sitting|One dog is sitting
Counting|Three people are waving|Three individuals wave their hands|One person is waving|Four people are waving
Action Confusion|A man is opening a door|A man pulls a door open|A man is closing a door|A man is knocking on a door
Action Confusion|A woman enters a room|A woman walks into a room|A woman leaves a room|A woman waits outside a room
Action Confusion|A boy throws a ball|A boy tosses a ball|A boy catches a ball|A boy holds a ball still
Action Confusion|A man sits down|A man takes a seat|A man stands up|A man remains standing
Action Confusion|A child climbs the stairs|A kid goes up the staircase|A child descends the stairs|A child sits on the stairs
Action Confusion|A woman picks up a cup|A woman lifts a cup|A woman puts down a cup|A woman washes a cup
Action Confusion|A dog jumps onto a couch|A dog leaps up onto a sofa|A dog jumps off a couch|A dog sleeps on a couch
Action Confusion|A person turns on a light|Someone switches a light on|Someone turns off a light|Someone replaces a light bulb
Attributes|A red car drives down the road|A crimson car is driving on a street|A blue car drives down the road|A green car drives down the road
Attributes|A small dog is running|A tiny dog is sprinting|A large dog is running|A medium-sized dog is running
Attributes|A person wears a black hat|Someone has a dark black hat on|A person wears a white hat|A person wears a red hat
Attributes|A woman holds a yellow umbrella|A woman carries a bright yellow umbrella|A woman holds a blue umbrella|A woman holds a pink umbrella
Attributes|A man rides a green bicycle|A man cycles on a green bike|A man rides a red bicycle|A man rides a blue bicycle
Attributes|A child carries a big box|A kid is carrying a large box|A child carries a small box|A child carries a tiny box
Attributes|A white cat is sleeping|A white-colored feline is asleep|A black cat is sleeping|A gray cat is sleeping
Attributes|A person holds a round balloon|Someone is holding a spherical balloon|A person holds a square balloon|A person holds a rectangular balloon
Compositionality|A dog runs while a cat sleeps|A cat is asleep while a dog is running|A dog sleeps while a cat runs|Both a dog and a cat are sleeping
Compositionality|A man eats while a woman dances|A woman dances as a man eats|A man dances while a woman eats|Both a man and a woman are eating
Compositionality|A boy laughs while a girl cries|A girl cries as a boy laughs|A boy cries while a girl laughs|Both the boy and girl laugh
Compositionality|A cat sits as a dog jumps|A dog jumps while a cat stays seated|A cat jumps as a dog sits|Both animals jump
Compositionality|A woman reads as a man cooks|A man prepares food while a woman reads|A woman cooks as a man reads|Both people read books
Compositionality|A child sleeps while an adult works|An adult works as a child sleeps|A child works while an adult sleeps|Both people are working
Compositionality|A dog swims while a bird flies|A bird flies as a dog swims|A dog flies while a bird swims|Both animals are swimming
Compositionality|A man waves while a woman claps|A woman applauds as a man waves|A man claps while a woman waves|Both people wave
Emotion|A man is laughing happily|A man laughs with joy|A man is crying sadly|A man looks upset and tearful
Emotion|A woman looks excited|A woman appears thrilled|A woman looks bored|A woman seems uninterested
Emotion|A child looks frightened|A child appears scared|A child looks confident and calm|A child looks delighted
Emotion|A person looks surprised|Someone appears astonished|Someone appears unsurprised|Someone looks bored
Emotion|A man looks angry|A man appears furious|A man looks pleased|A man appears joyful
Emotion|A girl feels proud|A girl looks pleased with her achievement|A girl feels ashamed|A girl looks embarrassed
Emotion|A dog looks relaxed|A dog appears calm and comfortable|A dog looks tense and alert|A dog appears frightened
Emotion|A person looks disappointed|Someone appears let down|Someone appears delighted|Someone celebrates excitedly
Synonyms|A person rides a bicycle|Someone is cycling|Someone drives a car|Someone pushes a bicycle
Synonyms|A man is laughing|A person is chuckling|A person is crying|A person is shouting angrily
Synonyms|A dog runs quickly|A canine sprints rapidly|A dog walks slowly|A dog lies down
Synonyms|A child is jumping|A kid is leaping|A child is sitting|A child is crawling
Synonyms|A woman is speaking|A lady is talking|A woman is listening silently|A woman is sleeping
Synonyms|A cat is sleeping|A feline is taking a nap|A cat is playing|A cat is jumping
Synonyms|A boy is dancing|A young boy is moving to the music|A boy is standing still|A boy is reading
Synonyms|A man is eating dinner|A person is having an evening meal|A man is cooking dinner|A man is washing dishes
Typo Robustness|A dog is running|A doog is runing|A dog is sleeping|A cat is running
Typo Robustness|A cat is sleeping|A catt is sleping|A cat is jumping|A dog is sleeping
Typo Robustness|A man is dancing|A maan is dancng|A man is sitting|A woman is dancing
Typo Robustness|A woman is laughing|A womn is lauging|A woman is crying|A man is laughing
Typo Robustness|A boy is jumping|A boi is jumpng|A boy is sleeping|A girl is jumping
Typo Robustness|A bird is flying|A brid is flyng|A bird is swimming|A plane is flying
Typo Robustness|A person is cycling|A persn is cyclng|A person is walking|A person is driving
Typo Robustness|A car is moving|A carr is moveing|A car is parked|A bike is moving
Multilingual|A dog is running|Một con chó đang chạy|A dog is sleeping|A cat is running
Multilingual|A cat is sleeping|Một con mèo đang ngủ|A cat is playing|A dog is sleeping
Multilingual|A man is laughing|Một người đàn ông đang cười|A man is crying|A woman is laughing
Multilingual|A woman is dancing|Một người phụ nữ đang nhảy múa|A woman is standing still|A man is dancing
Multilingual|A child is jumping|Một đứa trẻ đang nhảy lên|A child is sitting|An adult is jumping
Multilingual|A bird is flying|Một con chim đang bay|A bird is swimming|A plane is flying
Multilingual|A person rides a bicycle|Một người đang đi xe đạp|A person is driving a car|A person is walking
Multilingual|A dog is barking|Một con chó đang sủa|A dog is sleeping|A cat is meowing
'''

TEST_CASES = []
for line_number, line in enumerate(RAW.strip().splitlines(), 1):
    parts = [item.strip() for item in line.split('|')]
    if len(parts) != 5 or not all(parts):
        raise ValueError(f'Invalid test case on line {line_number}: {line}')
    category, query, positive, negative_a, negative_b = parts
    TEST_CASES.append({
        'id': line_number,
        'category': category,
        'query': query,
        'positive': positive,
        'negatives': [negative_a, negative_b],
    })
