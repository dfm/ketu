import turnstile

pipe = turnstile.Download()
pipe = turnstile.Inject(pipe)
pipe = turnstile.Prepare(pipe)
pipe = turnstile.Discontinuity(pipe)

q = {
    "kicid": 10453824,
}

r = pipe.query(**q)
