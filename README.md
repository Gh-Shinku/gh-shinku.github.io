# gh-shinku.github.io

## Blog
Record what i see and hear.

## Local development

Use the repository wrapper when serving or building the site:

```sh
./scripts/hugo-preprocess server
```

Omit `server` for a one-off build.

The wrapper preprocesses a temporary copy of `content/` before invoking Hugo.
It protects standalone `=` lines inside display-math blocks from being parsed
as Setext headings, while leaving the source Markdown unchanged. In server
mode, edits are synchronized into the temporary tree for live reload.

## Thanks
Use code from [hanwenguo/hugo-theme-nostyleplease](https://github.com/hanwenguo/hugo-theme-nostyleplease), a fork of this theme.
