library(MASS)
library(survival)
library(fitdistrplus)
library(readxl)
library(lmtest)

datos_df <- read_excel("D:/JUAN D QUIÑONES/Documents/Excel Tesis 1.xlsx", sheet = "OCCIDENTE-ORIENTE", range = "E1:E41")
datos <- datos_df$`t entre arribos final`

print(datos)
hist(datos, main = "Histograma de datos", xlab = "t entre arribos final")

ajusteexp <- fitdist(datos, "exp")
ajustegamma <- fitdist(datos, "gamma")
ajustenorm <- fitdist(datos, "norm")
ajustelnorm <- fitdist(datos, "lnorm")
ajusteunif <- fitdist(datos, "unif")
ajusteweibull <- fitdist(datos, "weibull")

resultadosexp <- gofstat(ajusteexp)
resultadosgamma <- gofstat(ajustegamma)
resultadosnorm <- gofstat(ajustenorm)
resultadoslnorm <- gofstat(ajustelnorm)
resultadosunif <- gofstat(ajusteunif)
resultadosweibull <- gofstat(ajusteweibull)

cat("P-Value Exponencial:", resultadosexp$chisqpvalue, "\n")
cat("P-Value Gamma:", resultadosgamma$chisqpvalue, "\n")
cat("P-Value Normal:", resultadosnorm$chisqpvalue, "\n")
cat("P-Value Lognormal:", resultadoslnorm$chisqpvalue, "\n")
cat("P-Value Uniforme:", resultadosunif$chisqpvalue, "\n")
cat("P-Value Weibull:", resultadosweibull$chisqpvalue, "\n")

plot(ajusteexp)
plot(ajustegamma)
plot(ajustenorm)
plot(ajustelnorm)
plot(ajusteunif)
plot(ajusteweibull)


