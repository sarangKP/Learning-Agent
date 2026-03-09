Here's a script designed to naturally arc from calm → confused → frustrated → calm, so you can watch all the config changes trigger:

---

**Phase 1 — Calm introduction (turns 1–4)**
```
1.  Hello, good morning. Who are you?
2.  Oh I see. Can you tell me what the weather is like today?
3.  That's nice. I had my tea already but I forgot if I took my tablet.
4.  I take one white tablet and one blue one every morning.
```

**Phase 2 — Confusion starts (turns 5–8)**
```
5.  I don't understand what you said. Can you say that again?
6.  I already asked you about my tablet. Did I take it or not?
7.  You're not making sense. I can't follow what you're telling me.
8.  I don't understand. Say it more simply please.
```

**Phase 3 — Frustration peak (turns 9–14)**
```
9.  I already told you about the tablet! Why do you keep asking me the same thing?
10. Nothing you say is helping me. I don't understand any of it.
11. I already asked this. You never remember what I tell you.
12. This is too complicated. I just want a simple answer.
13. I already told you! The white tablet and the blue tablet. Every morning!
14. You are making me very upset. Nothing is working.
```

**Phase 4 — Calming down (turns 15–17)**
```
15. Fine. Let's start over. Just tell me what I should do right now.
16. Okay that makes more sense. Thank you for being patient with me.
17. My knee is hurting a little today. Is that something to worry about?
```

**Phase 5 — Warm close (turns 18–20)**
```
18. I feel a bit lonely today. My daughter hasn't called in a while.
19. Can you tell me a short happy story? Something to cheer me up.
20. That was lovely. I think I'll rest now. Goodnight ELARA.
```

---

The key moments to watch for in the cyan panel:

- After turn **6–8** → expect `confused` detected, `clarity_level` drops to 1, `pace` goes to slow
- After turn **9–11** → expect `frustrated`, `patience_mode` flips to `True`, you'll hear ELARA start with empathy
- After turn **15–16** → affect should return to `calm`, config may ease back
- The **affect arc** at the end of the session will show the full journey