#data("USArrests")
#head(USArrests, 10)
datos <- read.csv("../datasets_simulados/educacion_v5.csv",dec=".",sep=";",header=TRUE)
head(datos,10)

#install.packages('corrplot')
library(corrplot)

# cada variable
summary(datos)
boxplot(datos[,1:6])
barplot(table(datos[,7]))

#relaciones entre variables
datos <- datos[,1:5]
datos <- datos[!is.na(datos[,1]),]
datos <- datos[!is.na(datos[,2]),]
datos <- datos[!is.na(datos[,3]),]
datos <- datos[!is.na(datos[,4]),]
datos <- datos[!is.na(datos[,5]),]
dim(datos)
summary(datos)
boxplot(datos)
plot(datos)
mcor<-cor(datos)
corrplot(mcor)
print(mcor)

# calcula varianzas para cada variable
apply(datos, 2, var)
apply(datos,2,mean)


# escalando los datos
scaled_df <- apply(datos, 2, scale)
head(scaled_df)
apply(scaled_df, 2, var)

apply(scaled_df, 2, mean)

summary(scaled_df)
boxplot(scaled_df)
pairs(scaled_df)
mcor<-cor(scaled_df)
corrplot(mcor)
print(mcor)

display_png(file="figura2.png")

# Calculando valores y vectores propios de la matriz de covarianzas empírica
mdat.cov <- cov(scaled_df)
mdat.eigen <- eigen(mdat.cov)
mdat.eigen


# Extrayendo los pesos de los dos primeras componentes principales 
w <- -mdat.eigen$vectors[,1:2] 
row.names(w) <- colnames(scaled_df)
colnames(w) <- c("PC1", "PC2")
w


# Calcula proyección de los datos en cada componente principal 
PC1 <- as.matrix(scaled_df) %*% w[,1]
PC2 <- as.matrix(scaled_df) %*% w[,2]

# Crea nuevo dataframe con la proyección
PC <- data.frame(PC1, PC2)
head(PC)

#install.packages('ggplot2')
library(ggplot2)
# Grafico en primer plano principal
ggplot(PC, aes(PC1, PC2)) + 
  modelr::geom_ref_line(h = 0) +
  modelr::geom_ref_line(v = 0) +
  geom_text(aes(label = 'o'), size = 3) +
  xlab("Primera Componente Principal") + 
  ylab("Segunda Componente Principal") + 
  ggtitle("Proyección en primer plano principal del los datos")

#Cálculo de la varianza explicada
PVE <- mdat.eigen$values / sum(mdat.eigen$values)
round(PVE, 2)


# Gráfico de la PVE
par(mfrow=c(1,2))
PVEplot <- barplot(PVE,xlab="Componente Principal", ylab="PVE", main= "Gráfico PVE", ylim=c(0, 1), names.arg=c(1:5))
cumPVE <- barplot(cumsum(PVE),xlab="Componente Principal", ylab="", main= "Gráfico PVE Acumulada", names.arg=c(1:5))


pca_res <- prcomp(datos, scale = TRUE)
names(pca_res)
pca_res$sdev
pca_res$rotation <- -pca_res$rotation
pca_res$center
pca_res$scale
pca_res$x <- -pca_res$x

biplot(pca_res, scale = 0,xlabs=rep('o',477))
