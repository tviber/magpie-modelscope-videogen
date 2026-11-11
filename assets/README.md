# Plugin icon

`modelscope-icon.png` is a square transparent, text-free adaptation of the
ModelScope emblem supplied by the user:
https://g.alicdn.com/sail-web/maas/2.13.182/static/logowithtextEn.005eb420.png

Prepared with the built-in image editing tool. The edit requested extraction of
the left purple/cyan emblem, removal of the lettering, and transparent padding
to a square canvas. This is a generated adaptation, not a pixel-exact crop.

The PNG is bundled for the provider icon and embedded as a data URI in
`package.json` for Magpie's plugin card, so both work offline. When replacing the
PNG, update `magpie.icon` to the same file's base64 data URI.
