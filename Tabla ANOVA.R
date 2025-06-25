library(readxl)

cola <- read_excel("Proyecto de grado.xlsx", sheet = "COLAMASLARGAB")
ppo <- read_excel("Proyecto de grado.xlsx", sheet = "PPOB")
a2c <- read_excel("Proyecto de grado.xlsx", sheet = "A2CB")

ppo <- ppo[244:293, ]
a2c <- a2c[812:861, ]

cola_tiempo <- cola$tiempo_prom_sistema
ppo_tiempo <- ppo$tiempo_prom_sistema
a2c_tiempo <- a2c$tiempo_prom_sistema

tiempos <- c(cola_tiempo, ppo_tiempo, a2c_tiempo)
grupo <- c(
  rep("COLA_MAS_LARGA", length(cola_tiempo)),
  rep("PPO", length(ppo_tiempo)),
  rep("A2C", length(a2c_tiempo))
)

df <- data.frame(tiempos, grupo)

anova_result <- aov(tiempos ~ grupo, data = df)
summary(anova_result)

