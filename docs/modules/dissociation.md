<script type="text/x-mathjax-config">
MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
MathJax.Ajax.config.path["mhchem"] =
  "https://cdnjs.cloudflare.com/ajax/libs/mathjax-mhchem/3.3.2";
MathJax.Hub.Config({TeX: {extensions: ["[mhchem]/mhchem.js"]}});
</script><script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML' async></script>

# Thermodynamic equilibrium constants

*The casual user has no need to explicitly call this module.*

`.dissociation` contains functions to evaluate thermodynamic equilibrium constants from temperature.

The functions all use the same syntax:

```python
lnk_EQ = pz.dissociation.EQ_SRC(tempK)
```

where `tempK` has its usual meanings, and `lnk_EQ` is the natural logarithm of the equilibrium constant for the equilibrium `EQ`, following `SRC`.

---

## Water

The thermodynamic dissociation constant $K(\ce{H2O})$ represents the equilibrium:
$$\ce{H2O <=> OH- + H+}$$
where:
$$K(\ce{H2O}) = \frac{ a(\ce{OH-}) a(\ce{H+})} {a(\ce{H2O})}$$

**Available functions:**

  * `.H2O_M88` - following [M88](../../references/#M88);
  * `.H2O_MF` - following [MF](../../references/#MF).

---

## Bisulfate dissociation

The thermodynamic dissociation constant $K(\ce{HSO4-})$ represents the equilibrium:
$$\ce{HSO4- <=> SO4^2- + H+}$$
where:
$$K(\ce{HSO4-}) = \frac{ a(\ce{SO4^2-}) a(\ce{H+})} {a(\ce{HSO4-})}$$

**Available functions:**

  * `HSO4_CRP94` - following [CRP94](../../references/#CRP94) Eq. 21;
  * `HSO4_CRP94_extra` - following [CRP94](../../references/#CRP94) Eq. 21, with additional digits on the constant term (S. L. Clegg, pers. comm., 7 February 2019).

---

## Magnesium hydroxide

The thermodynamic dissociation constant $K(\ce{Mg^2+})$ represents the equilibrium:
$$\ce{Mg^2+ + OH- <=> MgOH+}$$
where:
$$K(\ce{Mg^2+}) = \frac{ a(\ce{MgOH+})} {a(\ce{OH-}) a(\ce{Mg^2+})}$$

**Available functions:**

  * `Mg_CW91` - following [CW91](../../references/#CW91) Eq. 244 in log<sub>10</sub> and then converted to natural log;
  * `Mg_CW91_ln` - following [CW91](../../references/#CW91) Eq. 244.

---

## Tris buffer

The thermodynamic dissociation constant $K(\ce{trisH+})$ represents the equilibrium:
$$\ce{trisH+ <=> tris + H+}$$
where:
$$K(\ce{trisH+}) = \frac{ a(\ce{tris}) a(\ce{H+})} {a(\ce{trisH+})}$$

**Available function:**

  * `trisH_BH64` - following [BH64](../../references/#BH64) Eq. 3.
