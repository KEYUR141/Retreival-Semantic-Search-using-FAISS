import numpy as np
import json 
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# data = [""" 
         
#     Succession is an American satirical black comedy-drama television series created by Jesse Armstrong that aired for four seasons on HBO from June 3, 2018, to May 28, 2023. The series centers on the Roy family, the owners of global media and entertainment conglomerate Waystar RoyCo, and their fight for control of the company amidst uncertainty about the health of the family's patriarch.

#     Brian Cox portrays the family patriarch Logan Roy. His children are played by Alan Ruck as Connor, Jeremy Strong as Kendall, Kieran Culkin as Roman, and Sarah Snook as Shiv. Other starring cast members are Matthew Macfadyen as Tom Wambsgans, Shiv's husband and Waystar executive; Nicholas Braun as Greg Hirsch, Logan's grandnephew also employed by the company; Hiam Abbass as Marcia, Logan's third wife; and Peter Friedman as Frank Vernon, a longtime confidant of Logan; while Dagmara Domińczyk, Arian Moayed, J. Smith-Cameron, Justine Lupe, David Rasche, Fisher Stevens, and Alexander Skarsgård featured in recurring roles before being promoted to the main cast.

#     Succession received universal critical acclaim for its writing, acting, humor, musical score, directing, production values, and examination of its subject matter. Many critics and publications have named the show one of the greatest television series of all time.[8][9][10][11] The series has received several accolades, including three wins each for the Golden Globe for Best Television Series – Drama and the Primetime Emmy for Outstanding Drama Series in 2020, 2022 and 2023 as well as the British Academy Television Award for Best International Programme.[12] Culkin, Cox and Strong each won Golden Globe Award for Best Actor – Television Series Drama for their performances, and Culkin and Strong won the Primetime Emmy Award for Outstanding Lead Actor in a Drama Series. Snook and Macfadyen both also won Emmy Awards: for Lead Actress and Supporting Actor respectively, with Macfadyen winning twice. Armstrong also won four Emmys and a BAFTA for his writing.

#     Premise
#     Succession follows the Roy family, which owns the New York City-based global media conglomerate Waystar RoyCo. The family patriarch, Logan Roy, has experienced a decline in health. His four children—estranged oldest son Connor (Ruck), power-hungry Kendall (Strong), irreverent Roman (Culkin), and politically savvy Shiv (Snook)—who all have varying degrees of connection to the company begin to prepare for a future without their father and vie for prominence within the company.

#     Cast and characters
#     Main article: List of Succession characters
#     Hiam Abbass as Marcia Roy (seasons 1–2 and 4;[a] recurring season 3): Logan's third and current wife. Born and raised in Beirut, she is often at odds with Logan's mistrustful children. She has a son, Amir, from her first marriage and a daughter from a previous relationship.
#     Nicholas Braun as Greg Hirsch: the bumbling yet opportunistic grandson of Logan's brother Ewan. Greg is unfamiliar with the rough terrain he must navigate to win Logan over, and finds himself indentured to Tom Wambsgans in his quest for a place at Waystar.
#     Brian Cox as Logan Roy: the Dundee-born billionaire, born into poverty before establishing the media and entertainment conglomerate Waystar RoyCo. He is a brash leader whose primary focus is his company, rather than his four children: Connor from his first marriage and Kendall, Roman, and Siobhan from his second marriage. He is married to Marcia, his third wife.
#     Kieran Culkin as Roman Roy: half-brother to Connor and the middle child from Logan's second marriage. Roman is immature, does not take responsibilities seriously and often finds himself lacking the common sense his father requires of him. He is frequently at odds with his older brother Kendall and sometimes his sister Shiv, with whom he often vies for power and their father's attention.
#     Peter Friedman as Frank Vernon: COO and later vice-chairman of Waystar RoyCo, and longtime confidant of Logan. Frank is a member of Logan's old guard, on whom Kendall frequently relies to help win back Logan's favor. He is Kendall's mentor and godfather, and is disliked by Roman.
#     Natalie Gold as Rava Roy (season 1; recurring seasons 3–4): Kendall's former wife and mother of their two children.
#     Matthew Macfadyen as Tom Wambsgans: Shiv's fiancé and then her husband. Tom is a Waystar executive who is promoted from head of the amusement park and cruise division to running ATN, the company's global news outlet. He enjoys his proximity to the Roy family's power, but is frequently dismissed by the family's inner circle. He ingratiates himself with those more powerful than he, but torments his hapless subordinate Greg.
#     Alan Ruck as Connor Roy: the only child from Logan's first marriage. Mostly removed from corporate affairs, he defers to his half-siblings on most firm-related matters and resides at a ranch in New Mexico with his younger girlfriend Willa. He is prone to delusions of grandeur and "had an interest in politics from a young age." Similar to his half-siblings, Connor does not have the best recollections of his childhood, as he mentioned that he went three years without seeing Logan when he was a child.
#     Sarah Snook as Siobhan "Shiv" Roy: Logan's youngest child and only daughter. A left-leaning political fixer, she worked for a time for presidential candidate Gil Eavis, whose political views clash with Waystar. She eventually leaves politics to focus on building a future at Waystar. She is engaged to and then marries Tom, but their relationship is undermined by Shiv's infidelity.
#     Jeremy Strong as Kendall Roy: the younger half-brother of Connor and the eldest child from Logan's second marriage. As Logan's heir apparent, Kendall struggles to prove his worth to his father after botching major deals and battling substance abuse. He toils to maintain a relationship with his siblings, as well as his ex-wife Rava and their children.
#     Rob Yang as Lawrence Yee (seasons 1–2): the founder of media website Vaulter that is acquired by Waystar RoyCo. He holds great contempt for Waystar and particularly Kendall, with whom he is often at odds.
#     Dagmara Domińczyk as Karolina Novotney (seasons 2–4; recurring season 1): the head of public relations for Waystar RoyCo and a member of the company's legal team.
#     Arian Moayed as Stewy Hosseini (seasons 2 and 4;[a] recurring seasons 1 and 3): Kendall's friend from the Buckley School and Harvard who is now a private equity investor with a seat on Waystar's board. He is covertly in partnership with Logan's rival Sandy Furness.
#     J. Smith-Cameron as Gerri Kellman (seasons 2–4; recurring season 1): the general counsel to Waystar RoyCo, who is also godmother to Shiv. She becomes a mentor figure to Roman, with whom she also shares a secret sexual connection.
#     Justine Lupe as Willa Ferreyra (seasons 3–4; recurring seasons 1–2): Connor's girlfriend, and later wife, who is younger than him, is a former call girl and an aspiring playwright.[13]
#     David Rasche as Karl Muller (seasons 3–4; recurring seasons 1–2): Waystar RoyCo's CFO and member of the company's legal team.[13]
#     Fisher Stevens as Hugo Baker (seasons 3–4; recurring season 2): a senior communications executive for Parks and Cruises in charge of managing a scandal involving Brightstar cruise lines.[13]
#     Alexander Skarsgård as Lukas Matsson (season 4;[a] recurring season 3): the Swedish CEO of streaming media giant GoJo who is looking to buy Waystar RoyCo.[14]
#     """
# ]

data = [ """
    Succession, American comedy-drama television series created by British writer and producer Jesse Armstrong that aired on HBO from 2018 to 2023. The series focuses on the Roy family, whose aging patriarch, Logan Roy, owns the entertainment and media conglomerate Waystar Royco, one of the last surviving legacy media concerns, and struggles to pick a successor from among his power-hungry children, advisers, and investors. While Roy reluctantly acknowledges the need to choose a successor, he cannot seem to find one that satisfies both his desire to maintain family control over his company and to leave his life’s work to someone as mercilessly ambitious as he is.

    Widely praised for its imaginative profanity-laden dialogue, its prismatic classical music score by Nicholas Britell, and masterful performances by a cast of seasoned actors, Succession follows cutthroat corporate maneuverings and personal betrayals as almost everyone around Roy competes to succeed him.

    Cast and characters
    Succession
    SuccessionActors Jeremy Strong (left) and Alan Ruck portraying Kendall and Connor Roy in Succession (2018–23).
    Continuing HBO’s penchant for popularizing some of television’s most notorious antiheroes (such as in The Sopranos, The Wire, and Game of Thrones), Succession is a show with virtually no “good guys.” The series has garnered praise for its nuanced characters, who are deeply flawed, often selfish and cruel, and yet not completely unsympathetic to viewers.

    Iron-fisted billionaire Logan Roy, played by Scottish actor Brian Cox, has raised his four children within family dynamics defined by extravagant wealth, scarcity of affection, and constant competition. Having grown up poor and survived familial abuse throughout his childhood in Scotland only to move to the United States and become one of the country’s most powerful and influential individuals, Logan sought to give his children the comfort and amenities he never had and to raise them to be as hard-boiled and resourceful as he is. At the start of the series, as Logan turns 80 years old, he seems utterly unsatisfied with the results of his parenting.

    His children desperately seek his approval, which mires them in the paradox that Logan would never give his respect to anyone so desirous of it. His eldest son, Connor Roy, played by American actor Alan Ruck, attempts to stay out of the fray, on his ranch in New Mexico, consumed by his comical Libertarian-leaning U.S. presidential run. Although he appears in some ways to be the natural choice to assume control of Waystar, Kendall Roy, Logan’s first child from his second marriage, played by American actor Jeremy Strong, also suffers from delusions of grandeur, as he appears to consider himself a more cunning and capable corporate operative than he proves to be. He struggles with substance use disorder, the aftermath of a painful divorce, and his father’s apparent low regard for him, but he persists in his efforts to become the successor.

    Succession
    SuccessionNicholas Braun (left) and Matthew Macfadyen in the television series Succession (2018–23).
    Siobhan (“Shiv”) Roy, played by Australian actress Sarah Snook, at first seems driven to find success outside the family confines as a political consultant but eventually owns up to her desire to take the reins at Waystar. She keeps everyone, including her husband, at an emotional distance. Logan’s youngest child, Roman Roy, played by American actor Kieran Culkin, seems to be the underdog in the filial power struggle. Sarcastic and vulgar, Roman is terrified of vulnerability and seems to lack the courage to defy his father. The family is rounded out by Shiv’s obsequious, social-climbing husband, Tom Wambsgans, a Midwestern transplant played by British actor Matthew Macfadyen, and the Roy children’s cousin Greg Hirsch, who is alternatingly a bumbling fool and an enterprising sycophant, portrayed by American actor Nicholas Braun.

    Despite their lavish wealth, the Roys’ world often appears unmistakably joyless, which is reflected in the show’s sterile, impersonal settings and bland design choices. Although the Roys derive such little pleasure or satisfaction from their wealth, “For these people to be excluded from the flame of money and power, I think, would feel a bit like death,” creator Armstrong told The New Yorker magazine in 2021.


    Access for the whole family!
    Bundle Britannica Premium and Kids for the ultimate resource destination.
    Real-life inspiration
    Napoleonic succession, Succession style
    Napoleonic succession, Succession styleNapoleon I, emperor of the French, is one of the most celebrated personages in the history of the West.
    See all videos for this article
    Although Armstrong claims that his portrait of the Roys was influenced by many different famous media family dynasties, including the Redstones, the Hearsts, and the Mercers, much attention has been paid to the similarities between Logan Roy’s family and that of Rupert Murdoch, the Australian-born American media mogul and founder of Fox News.

    In a 2023 article for The Guardian, Armstrong described some of the characteristics and personality traits shared by the media moguls after whom he modeled Logan Roy: “They were connected by a strong interest in a few things: a refusal to think about mortality…; desire for control; manic deal-making energy; love of gossip and power-connection; a certain ruthlessness about hirings and firings. And most of all, an instinct for forward motion, with a notable lack of introspection.”

    Critical reception
    Although Succession has drawn fewer viewers than HBO’s most popular shows, it has received extensive critical attention from national outlets covering news and culture. In 2023 reporting on media trends found that Succession spawned six times as many online articles in one 30-day period in summer 2023 as any other highly watched television show and seemed to confirm that Succession sparked outsized media coverage compared with reader interest. Some critics have speculated that the media’s fascination with the show stems from its own involvement in the volatile and sometimes toxic media industry that the show portrays.

    """
]
lines = data[0].splitlines()
texts =  [line.strip() for line in lines if line.strip()]

embeddings = model.encode(texts)
np.save("embeddings.npy",embeddings)


with open("texts.json", "w", encoding="utf-8") as f:
    json.dump(texts, f, ensure_ascii=False, indent=4)


