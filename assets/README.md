# assets

`muser-book-social-card.png` is the 1200×630 repository social preview.
The generator writes the identical served copy to
`src/muser-book-social-card.png`; mdBook publishes that copy with every build.
Do not edit either file by hand:

```sh
python3 scripts/generate_social_card.py           # regenerate
python3 scripts/generate_social_card.py --check   # verify committed copy
```

The GitHub repository preview is configured manually: repo → Settings →
General → Social preview → upload the `assets/` copy. Open Graph and X cards
use the served copy automatically.
