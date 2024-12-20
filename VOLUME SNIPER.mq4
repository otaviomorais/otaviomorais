#property copyright "Copyright 2024"
#include <WinUser32.mqh>
#import "user32.dll"
int RegisterWindowMessageA(string a0);
#import
#property link      ""
#property version   "1.00"
#property strict
#property indicator_chart_window
#property indicator_buffers 2
#property indicator_plots   2

//--- plot Buy Signal
#property indicator_label1  "Buy Signal"
#property indicator_type1   DRAW_ARROW
#property indicator_color1  clrLime
#property indicator_style1  STYLE_SOLID
#property indicator_width1  1

//--- plot Sell Signal
#property indicator_label2  "Sell Signal"
#property indicator_type2   DRAW_ARROW
#property indicator_color2  clrRed 
#property indicator_style2  STYLE_SOLID
#property indicator_width2  1

extern int ArrowDistance = 10;          // Distância das setas em pontos
extern bool EnableAlerts = false;        // Habilitar alertas sonoros
 bool EnablePushNotifications = false; // Habilitar notificações push
 int MinimumTouchDistance = 4;    // Distância mínima entre toques em barras
 int MinimumTimeSeconds = 300;    // Tempo mínimo entre sinais (5 minutos)
 int TrendLineTouchSensitivity = 5; // Sensibilidade do toque na linha de tendência
 int ExpirationBars = 1;          // Quantidade de barras para expiração
 int MomentumBars = 3;            // Barras para verificar momentum

//--- Volume Profile parameters
 int    BarrasAnalise = 500;     // Barras para análise do Volume Profile
 int    PrecisionPrice = 90;     // Precisão do preço para Volume Profile
 int    ValueAreaPercent = 90;    // Percentual da Value Area
 double VolumeRatio = 1.5;        // Razão de volume para confirmação
extern bool   UseVolumeFilter = true;   // Usar filtro de Volume Profile
 color  UpColor = clrSteelBlue;   // Cor das barras de compra
 color  DownColor = clrChocolate; // Cor das barras de venda
 color  POCColor = clrYellow;     // Cor do POC
 color  VAHColor = clrWhite;      // Cor do Value Area High
 color  VALColor = clrWhite;      // Cor do Value Area Low
 int    BarWidth = 60;            // Largura máxima das barras
 bool   ShowLabels = true;        // Mostrar labels com valores
 bool Auto_Refresh = TRUE;
 int Normal_TL_Period = 500;
 bool Three_Touch = TRUE;
 bool M1_Fast_Analysis = TRUE;
 bool M5_Fast_Analysis = TRUE;
 bool Mark_Highest_and_Lowest_TL = TRUE;
 int Expiration_Day_Alert = 5;
 color Normal_TL_Color = Gainsboro;
 color Long_TL_Color = Goldenrod;
 int Three_Touch_TL_Widht = 2;
 color Three_Touch_TL_Color = White;

int gi_120;
int gi_124;


//--- indicator buffers
double BuyBuffer[];
double SellBuffer[];
   double ld_20;
   double ld_28;
   double ld_36;
   double ld_44;
   double ld_52;
   double ld_60;
   double ld_68;
   double ld_76;
   double ld_84;
   double ld_100;
   double ld_108;
   double ld_116;
   double ld_124;
   double ld_132;
   double ld_140;
   double ld_148;
   double ld_156;
   double ld_164;
   double ld_172;
   double ld_180;
   double ld_188;
   double ld_232;
   double ld_240;
   
   int li_248;
   int li_252;
   int li_200;
   int li_204;
   int li_208;
   int li_212;
   int li_216;
   int li_220;

// Estrutura para Volume Profile
struct VOLUME_LEVEL {
    double price;
    double buyVolume;
    double sellVolume;
    double totalVolume;
    double volumePercent;
    bool isInValueArea;
};

VOLUME_LEVEL volumeLevels[];
int pocLevel = -1;
int vahLevel = -1;
int valLevel = -1;
datetime lastAlertTime = 0;
datetime lastSignalTime = 0;
int alertMinimumInterval = 60;

//+------------------------------------------------------------------+
//| Custom indicator initialization function                           |
//+------------------------------------------------------------------+
int OnInit()
{
    SetIndexBuffer(0, BuyBuffer);
    SetIndexBuffer(1, SellBuffer);
    
    SetIndexArrow(0, 233);
    SetIndexArrow(1, 234);
    
    ArrayInitialize(BuyBuffer, EMPTY_VALUE);
    ArrayInitialize(SellBuffer, EMPTY_VALUE);
    
    PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, EMPTY_VALUE);
    PlotIndexSetDouble(1, PLOT_EMPTY_VALUE, EMPTY_VALUE);
    
   ObjectCreate("calctl", OBJ_HLINE, 0, 0, 0);
   ObjectCreate("visibletl", OBJ_HLINE, 0, 0, 0);
   ObjectCreate("downmax", OBJ_TREND, 0, 0, 0, 0, 0);
   ObjectCreate("upmax", OBJ_TREND, 0, 0, 0, 0, 0);

    
    return(INIT_SUCCEEDED);
}

int deinit() {
   for (int li_0 = 0; li_0 <= 100; li_0++) {
      ObjectDelete("downtrendline" + li_0);
      ObjectDelete("uptrendline" + li_0);
      ObjectDelete("downtrendline" + li_0 + "tt");
      ObjectDelete("uptrendline" + li_0 + "tt");
   }
   ObjectDelete("calctl");
   ObjectDelete("timeleft");
   ObjectDelete("invacc");
   ObjectDelete("visibletl");
   ObjectDelete("downmax");
   ObjectDelete("upmax");
   ObjectDelete("downmax");
   ObjectDelete("upmax");
   return (0);
}

//+------------------------------------------------------------------+
//| Sort levels by volume                                             |
//+------------------------------------------------------------------+
void SortLevelsByVolume(VOLUME_LEVEL &levels[], int size)
{
    for(int i = 0; i < size - 1; i++)
    {
        for(int j = i + 1; j < size; j++)
        {
            if(levels[j].totalVolume > levels[i].totalVolume)
            {
                VOLUME_LEVEL temp = levels[i];
                levels[i] = levels[j];
                levels[j] = temp;
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Calculate Volume Profile levels                                    |
//+------------------------------------------------------------------+
void CalculateVolumeProfile()
{
    // Limpa objetos anteriores
    ObjectsDeleteAll(0, "VProfile_");
    
    // Calcula níveis de preço
    double maxPrice = High[iHighest(NULL, 0, MODE_HIGH, BarrasAnalise, 0)];
    double minPrice = Low[iLowest(NULL, 0, MODE_LOW, BarrasAnalise, 0)];
    double increment = NormalizeDouble((maxPrice - minPrice) / PrecisionPrice, _Digits);
    
    ArrayResize(volumeLevels, PrecisionPrice);
    
    // Inicializa níveis
    for(int i = 0; i < PrecisionPrice; i++)
    {
        volumeLevels[i].price = NormalizeDouble(minPrice + (i * increment), _Digits);
        volumeLevels[i].buyVolume = 0;
        volumeLevels[i].sellVolume = 0;
        volumeLevels[i].totalVolume = 0;
        volumeLevels[i].volumePercent = 0;
        volumeLevels[i].isInValueArea = false;
    }
    
    // Calcula volumes
    double totalVolumeAll = 0;
    for(int i = 0; i < BarrasAnalise; i++)
    {
        double barVolume = (double)Volume[i];
        totalVolumeAll += barVolume;
        
        bool isBuyBar = Close[i] > Open[i];
        
        for(int j = 0; j < PrecisionPrice; j++)
        {
            if(Close[i] >= volumeLevels[j].price && Close[i] < volumeLevels[j].price + increment)
            {
                if(isBuyBar)
                    volumeLevels[j].buyVolume += barVolume;
                else
                    volumeLevels[j].sellVolume += barVolume;
                    
                volumeLevels[j].totalVolume += barVolume;
                break;
            }
        }
    }
    
    // Calcula percentuais e encontra POC
    double maxVolume = 0;
    for(int i = 0; i < PrecisionPrice; i++)
    {
        volumeLevels[i].volumePercent = (volumeLevels[i].totalVolume / totalVolumeAll) * 100;
        
        if(volumeLevels[i].totalVolume > maxVolume)
        {
            maxVolume = volumeLevels[i].totalVolume;
            pocLevel = i;
        }
    }
    
    // Calcula Value Area
    double totalVolume = 0;
    double targetVolume = totalVolumeAll * ValueAreaPercent / 100.0;
    
    // Cria cópia ordenada para Value Area
    VOLUME_LEVEL tempLevels[];
    ArrayResize(tempLevels, PrecisionPrice);
    ArrayCopy(tempLevels, volumeLevels);
    SortLevelsByVolume(tempLevels, PrecisionPrice);
    
    // Marca níveis na Value Area
    for(int i = 0; i < PrecisionPrice && totalVolume < targetVolume; i++)
    {
        for(int j = 0; j < PrecisionPrice; j++)
        {
            if(volumeLevels[j].totalVolume == tempLevels[i].totalVolume)
            {
                volumeLevels[j].isInValueArea = true;
                totalVolume += volumeLevels[j].totalVolume;
                break;
            }
        }
    }
    
    // Encontra VAH e VAL
    vahLevel = -1;
    valLevel = PrecisionPrice;
    for(int i = 0; i < PrecisionPrice; i++)
    {
        if(volumeLevels[i].isInValueArea)
        {
            if(vahLevel == -1) vahLevel = i;
            valLevel = i;
        }
    }
    
    // Desenha barras e labels
    datetime rightEdge = Time[0] + PeriodSeconds() * 20;
    datetime leftEdge = Time[0];
    
    for(int i = 0; i < PrecisionPrice; i++)
    {
        if(volumeLevels[i].totalVolume == 0) continue;
        
        string buyName = "VProfile_Buy_" + IntegerToString(i);
        string sellName = "VProfile_Sell_" + IntegerToString(i);
        string labelName = "VProfile_Label_" + IntegerToString(i);
        
        double buyWidth = (volumeLevels[i].buyVolume / maxVolume) * BarWidth;
        double sellWidth = (volumeLevels[i].sellVolume / maxVolume) * BarWidth;
        
        // Desenha barra de compra
        if(volumeLevels[i].buyVolume > 0)
        {
            ObjectCreate(0, buyName, OBJ_RECTANGLE, 0, 
                leftEdge, volumeLevels[i].price,
                leftEdge + (buyWidth * PeriodSeconds()), volumeLevels[i].price + increment);
            
            ObjectSetInteger(0, buyName, OBJPROP_COLOR, UpColor);
            ObjectSetInteger(0, buyName, OBJPROP_FILL, true);
            ObjectSetInteger(0, buyName, OBJPROP_BACK, true);
        }
        
        // Desenha barra de venda
        if(volumeLevels[i].sellVolume > 0)
        {
            ObjectCreate(0, sellName, OBJ_RECTANGLE, 0, 
                leftEdge - (sellWidth * PeriodSeconds()), volumeLevels[i].price,
                leftEdge, volumeLevels[i].price + increment);
            
            ObjectSetInteger(0, sellName, OBJPROP_COLOR, DownColor);
            ObjectSetInteger(0, sellName, OBJPROP_FILL, true);
            ObjectSetInteger(0, sellName, OBJPROP_BACK, true);
        }
        
        // Adiciona labels
        if(ShowLabels && volumeLevels[i].volumePercent >= 1.0)
        {
            ObjectCreate(0, labelName, OBJ_TEXT, 0,
                rightEdge, volumeLevels[i].price + (increment/2));
            ObjectSetString(0, labelName, OBJPROP_TEXT, 
                DoubleToString(volumeLevels[i].volumePercent, 1) + "%");
            ObjectSetInteger(0, labelName, OBJPROP_COLOR, clrWhite);
            ObjectSetInteger(0, labelName, OBJPROP_BACK, false);
        }
    }
    
    // Desenha POC, VAH e VAL
    if(pocLevel >= 0)
    {
        string pocName = "VProfile_POC";
        ObjectCreate(0, pocName, OBJ_TREND, 0,
            leftEdge - (BarWidth * PeriodSeconds()), volumeLevels[pocLevel].price,
            rightEdge, volumeLevels[pocLevel].price);
        ObjectSetInteger(0, pocName, OBJPROP_COLOR, POCColor);
        ObjectSetInteger(0, pocName, OBJPROP_STYLE, STYLE_DASH);
        ObjectSetInteger(0, pocName, OBJPROP_WIDTH, 2);
    }
    
    if(vahLevel >= 0)
    {
        string vahName = "VProfile_VAH";
        ObjectCreate(0, vahName, OBJ_TREND, 0,
            leftEdge - (BarWidth * PeriodSeconds()), volumeLevels[vahLevel].price,
            rightEdge, volumeLevels[vahLevel].price);
        ObjectSetInteger(0, vahName, OBJPROP_COLOR, VAHColor);
        ObjectSetInteger(0, vahName, OBJPROP_STYLE, STYLE_DOT);
    }
    
    if(valLevel < PrecisionPrice)
    {
        string valName = "VProfile_VAL";
        ObjectCreate(0, valName, OBJ_TREND, 0,
            leftEdge - (BarWidth * PeriodSeconds()), volumeLevels[valLevel].price,
            rightEdge, volumeLevels[valLevel].price);
        ObjectSetInteger(0, valName, OBJPROP_COLOR, VALColor);
        ObjectSetInteger(0, valName, OBJPROP_STYLE, STYLE_DOT);
    }
}

//+------------------------------------------------------------------+
//| Melhorada: Verifica Volume Profile considerando range de preços   |
//+------------------------------------------------------------------+
bool IsVolumeProfileConfirming(int bar, bool isBuySignal)
{
    if(!UseVolumeFilter) return true;
    
    double highPrice = High[bar];
    double lowPrice = Low[bar];
    
    double totalBuyVolume = 0;
    double totalSellVolume = 0;
    
    // Verifica volume em toda a faixa de preço da barra
    for(int i = 0; i < PrecisionPrice; i++)
    {
        if((lowPrice <= volumeLevels[i].price && highPrice >= volumeLevels[i].price))
        {
            totalBuyVolume += volumeLevels[i].buyVolume;
            totalSellVolume += volumeLevels[i].sellVolume;
        }
    }
    
    if(isBuySignal)
        return totalBuyVolume > totalSellVolume * VolumeRatio;
    else
        return totalSellVolume > totalBuyVolume * VolumeRatio;
}

//+------------------------------------------------------------------+
//| Nova: Verifica momentum do movimento                              |
//+------------------------------------------------------------------+
bool HasMomentum(int bar, bool isBuySignal)
{
    int consecutiveBars = 0;
    
    for(int i = bar; i < bar + MomentumBars && i < Bars; i++)
    {
        if(isBuySignal)
        {
            if(Close[i] > Open[i])
                consecutiveBars++;
        }
        else
        {
            if(Close[i] < Open[i])
                consecutiveBars++;
        }
    }
    
    return (consecutiveBars >= MomentumBars/2);  // Pelo menos metade das barras na direção
}

//+------------------------------------------------------------------+
//| Nova: Verifica tempo mínimo entre sinais                          |
//+------------------------------------------------------------------+
bool CheckTimeDistance(datetime currentTime)
{
    if(currentTime - lastSignalTime < MinimumTimeSeconds) 
        return false;
    return true;
}

//+------------------------------------------------------------------+
//| Melhorada: Verifica toque na linha de tendência                   |
//+------------------------------------------------------------------+
bool IsTrendLineTouch(string lineName, int bar, bool isUpTrend)
{
    if(ObjectFind(0, lineName) < 0) return false;
    
    double lineValue = ObjectGetValueByTime(0, lineName, Time[bar]);
    if(lineValue <= 0) return false;
    
    double touchSensitivity = TrendLineTouchSensitivity * Point;
    
    if(isUpTrend)
    {
        return (Low[bar] <= lineValue + touchSensitivity && 
                Low[bar] >= lineValue - touchSensitivity &&
                Close[bar] > Open[bar]);  // Confirmação adicional
    }
    else
    {
        return (High[bar] >= lineValue - touchSensitivity && 
                High[bar] <= lineValue + touchSensitivity &&
                Close[bar] < Open[bar]);  // Confirmação adicional
    }
}

//+------------------------------------------------------------------+
//| Melhorada: Verifica confirmação da direção após o toque          |
//+------------------------------------------------------------------+
bool IsDirectionConfirmed(int bar, bool isBuySignal)
{
    if(bar >= Bars - 1) return false;
    
    if(isBuySignal)
    {
        return Close[bar] > Open[bar] && 
               Close[bar] > Close[bar+1] &&
               High[bar] > High[bar+1];  // Confirmação adicional
    }
    else
    {
        return Close[bar] < Open[bar] && 
               Close[bar] < Close[bar+1] &&
               Low[bar] < Low[bar+1];    // Confirmação adicional
    }
}


//+------------------------------------------------------------------+
//| Verifica se a linha é válida para a estratégia                    |
//+------------------------------------------------------------------+
bool IsValidTrendLine(string lineName)
{
    if(ObjectFind(0, lineName) < 0) return false;
    
    string desc = ObjectGetString(0, lineName, OBJPROP_TEXT);
    return StringFind(desc, "Long") >= 0 || 
           StringFind(desc, "Normal") >= 0 || 
           StringFind(desc, "3t") >= 0;
}

//+------------------------------------------------------------------+
//| Melhorada: Verifica distância entre toques                        |
//+------------------------------------------------------------------+
bool CheckTouchDistance(string lineName, int currentBar)
{
    if(currentBar >= Bars - MinimumTouchDistance) return false;
    
    bool isUpTrend = (StringFind(lineName, "uptrendline") >= 0);
    
    for(int i = currentBar + 1; i < currentBar + MinimumTouchDistance; i++)
    {
        if(IsTrendLineTouch(lineName, i, isUpTrend))
        {
            return false;
        }
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Melhorada: Verifica rompimento da linha de tendência             |
//+------------------------------------------------------------------+
bool IsTrendLineBroken(string lineName, int bar, bool isUpTrend)
{
    if(ObjectFind(0, lineName) < 0) return false;

    double lineValue = ObjectGetValueByTime(0, lineName, Time[bar]);
    if(lineValue <= 0) return false;

    // Verifica múltiplas barras para confirmar rompimento
    int confirmationBars = 2;
    int brokenBars = 0;

    for(int i = bar; i < bar + confirmationBars && i < Bars; i++)
    {
        if(isUpTrend)
        {
            if(Close[i] < lineValue && Low[i] < lineValue)
                brokenBars++;
        }
        else
        {
            if(Close[i] > lineValue && High[i] > lineValue)
                brokenBars++;
        }
    }

    return (brokenBars >= confirmationBars);
}

//+------------------------------------------------------------------+
//| Envia alertas                                                     |
//+------------------------------------------------------------------+
void SendAlerts(string direction)
{
    datetime currentTime = TimeCurrent();
    if(currentTime - lastAlertTime < alertMinimumInterval) return;
    
    string message = StringFormat("Sinal de %s detectado!", direction);
    
    Alert(message);
    PlaySound("alert.wav");
    
    if(EnablePushNotifications)
    {
        SendNotification(message);
    }
    
    lastAlertTime = currentTime;
}

//+------------------------------------------------------------------+
//| Custom indicator iteration function                                |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
{

   if (Normal_TL_Period > 1000 || Normal_TL_Period < 100) Normal_TL_Period = 500;
   string ls_0 = AccountNumber();
   gi_124++;
   string ls_8 = AccountNumber();
   int li_16 = 1;
   int li_196 = MathMax(0, WindowFirstVisibleBar() - WindowBarsPerChart());
   double ld_224 = Bars;
   if (gi_120 == 0) gi_120 = ld_224;
   if (ld_224 > gi_120) {
      gi_120 = ld_224;
      if (Auto_Refresh == TRUE && li_196 == 0) ObjectSet("calctl", OBJPROP_PRICE1, -1);
   }
   if (Auto_Refresh == TRUE && IndicatorCounted() == 0) ObjectSet("calctl", OBJPROP_PRICE1, -1);
   if (ObjectGet("visibletl", OBJPROP_PRICE1) == -1.0) {
      for (int li_208 = 0; li_208 <= 100; li_208++) {
         ObjectDelete("downtrendline" + li_208);
         ObjectDelete("uptrendline" + li_208);
         ObjectDelete("downtrendline" + li_208 + "tt");
         ObjectDelete("uptrendline" + li_208 + "tt");
      }
   }
   if (ObjectGet("calctl", OBJPROP_PRICE1) == -1.0 && ObjectGet("visibletl", OBJPROP_PRICE1) == 0.0 && StringFind(ls_8, ls_0, 0) >= 0 && li_16 > 0) {
      for (li_208 = 0; li_208 <= 100; li_208++) {
         ObjectDelete("downtrendline" + li_208);
         ObjectDelete("uptrendline" + li_208);
         ObjectDelete("downtrendline" + li_208 + "tt");
         ObjectDelete("uptrendline" + li_208 + "tt");
      }
      ld_20 = 150000;
      if (Period() == PERIOD_M1 && M1_Fast_Analysis == TRUE) ld_20 = 8000;
      if (Period() == PERIOD_M5 && M5_Fast_Analysis == TRUE) ld_20 = 2400;
      if (Period() == PERIOD_MN1) {
         ld_20 = 150;
         Three_Touch = FALSE;
         Normal_TL_Period = 150;
      }
      ld_28 = li_196 + MathMin(Bars - li_196 - 10, ld_20);
      ld_36 = iHigh(NULL, 0, ld_28);
      ld_52 = li_196 + MathMin(Bars - li_196 - 10, ld_20);
      ld_60 = iHigh(NULL, 0, ld_52);
      for (int li_200 = 1; li_200 < 50; li_200++) {
         if ((iFractals(NULL, 0, MODE_UPPER, li_196 + li_200) > 0.0 && li_200 > 2) || (Close[li_196 + li_200 + 1] > Open[li_196 + li_200 + 1] && Close[li_196 + li_200 + 1] - (Low[li_196 +
            li_200 + 1]) < 0.6 * (High[li_196 + li_200 + 1] - (Low[li_196 + li_200 + 1])) && Close[li_196 + li_200] < Open[li_196 + li_200]) || (Close[li_196 + li_200 + 1] <= Open[li_196 +
            li_200 + 1] && Close[li_196 + li_200] < Open[li_196 + li_200]) || (Close[li_196 + li_200] < Open[li_196 + li_200] && Close[li_196 + li_200] < Low[li_196 + li_200 + 1])) {
            ld_44 = li_196 + li_200;
            break;
         }
      }
      for (int li_204 = 1; li_204 <= 30; li_204++) {
         if (ld_28 > ld_44 + 6.0) {
            ObjectCreate("downtrendline" + li_204, OBJ_TREND, 0, iTime(NULL, 0, ld_28), ld_36, iTime(NULL, 0, ld_28), ld_36);
            for (li_200 = ld_28; li_200 >= ld_44; li_200--) {
               if (ObjectGet("downtrendline" + li_204, OBJPROP_PRICE1) == ObjectGet("downtrendline" + li_204, OBJPROP_PRICE2)) {
                  ObjectMove("downtrendline" + li_204, 1, iTime(NULL, 0, li_200 - 1), iHigh(NULL, 0, li_200 - 1));
                  ld_28 = li_200 - 1;
                  ld_36 = iHigh(NULL, 0, li_200 - 1);
               }
               ld_76 = ObjectGetValueByShift("downtrendline" + li_204, li_200);
               if (ld_76 < iHigh(NULL, 0, li_200)) {
                  ObjectMove("downtrendline" + li_204, 1, iTime(NULL, 0, li_200), iHigh(NULL, 0, li_200));
                  ld_28 = li_200;
                  ld_36 = iHigh(NULL, 0, li_200);
               }
            }
         }
         if (ObjectGet("downtrendline" + li_204, OBJPROP_PRICE1) < ObjectGet("downtrendline" + li_204, OBJPROP_PRICE2)) ObjectDelete("downtrendline" + li_204);
         if (iBarShift(NULL, 0, ObjectGet("downtrendline" + li_204, OBJPROP_TIME1)) - li_196 >= Normal_TL_Period) {
            ObjectSet("downtrendline" + li_204, OBJPROP_COLOR, Long_TL_Color);
            ObjectSetText("downtrendline" + li_204, "Long");
         } else {
            ObjectSet("downtrendline" + li_204, OBJPROP_COLOR, Normal_TL_Color);
            ObjectSetText("downtrendline" + li_204, "Normal");
         }
      }
      for (li_200 = 1; li_200 < 50; li_200++) {
         if ((iFractals(NULL, 0, MODE_LOWER, li_196 + li_200) > 0.0 && li_200 > 2) || (Close[li_196 + li_200 + 1] < Open[li_196 + li_200 + 1] && High[li_196 + li_200 + 1] - (Close[li_196 +
            li_200 + 1]) < 0.6 * (High[li_196 + li_200 + 1] - (Low[li_196 + li_200 + 1])) && Close[li_196 + li_200] > Open[li_196 + li_200]) || (Close[li_196 + li_200 + 1] >= Open[li_196 +
            li_200 + 1] && Close[li_196 + li_200] > Open[li_196 + li_200]) || (Close[li_196 + li_200] > Open[li_196 + li_200] && Close[li_196 + li_200] > High[li_196 + li_200 + 1])) {
            ld_68 = li_196 + li_200;
            break;
         }
      }
      for (li_204 = 1; li_204 <= 30; li_204++) {
         if (ld_52 > ld_68 + 6.0) {
            ObjectCreate("uptrendline" + li_204, OBJ_TREND, 0, iTime(NULL, 0, ld_52), ld_60, iTime(NULL, 0, ld_52), ld_60);
            for (li_200 = ld_52; li_200 >= ld_68; li_200--) {
               if (ObjectGet("uptrendline" + li_204, OBJPROP_TIME1) == ObjectGet("uptrendline" + li_204, OBJPROP_TIME2)) {
                  ObjectMove("uptrendline" + li_204, 1, iTime(NULL, 0, li_200 - 1), iLow(NULL, 0, li_200 - 1));
                  ld_52 = li_200 - 1;
                  ld_60 = iLow(NULL, 0, li_200 - 1);
               }
               ld_76 = ObjectGetValueByShift("uptrendline" + li_204, li_200);
               if (iLow(NULL, 0, li_200) < ld_76) {
                  ObjectMove("uptrendline" + li_204, 1, iTime(NULL, 0, li_200), iLow(NULL, 0, li_200));
                  ld_52 = li_200;
                  ld_60 = iLow(NULL, 0, li_200);
               }
            }
         }
         if (ObjectGet("uptrendline" + li_204, OBJPROP_PRICE1) > ObjectGet("uptrendline" + li_204, OBJPROP_PRICE2)) ObjectDelete("uptrendline" + li_204);
         if (iBarShift(NULL, 0, ObjectGet("uptrendline" + li_204, OBJPROP_TIME1)) - li_196 >= Normal_TL_Period) {
            ObjectSet("uptrendline" + li_204, OBJPROP_COLOR, Long_TL_Color);
            ObjectSetText("uptrendline" + li_204, "Long");
         } else {
            ObjectSet("uptrendline" + li_204, OBJPROP_COLOR, Normal_TL_Color);
            ObjectSetText("uptrendline" + li_204, "Normal");
         }
      }
      if (Three_Touch == TRUE && Bars > 1000) {
         for (li_204 = 1; li_204 <= 30; li_204++) {
            ld_100 = ObjectGet("downtrendline" + li_204, OBJPROP_TIME1);
            ld_108 = iBarShift(NULL, 0, ld_100);
            ld_84 = ld_44;
            ld_116 = ld_108 - ld_84;
            if (ld_116 < MathMin(Normal_TL_Period, 1000) && ld_116 > 6.0) {
               ObjectCreate("downtrendline" + li_204 + "tt", OBJ_TREND, 0, iTime(NULL, 0, ld_108), iHigh(NULL, 0, ld_108), iTime(NULL, 0, ld_84), iHigh(NULL, 0, ld_84));
               ObjectSet("downtrendline" + li_204 + "tt", OBJPROP_WIDTH, 2);
               ld_180 = iATR(NULL, 0, ld_116, li_196) / Point / 10.0;
               ld_188 = 8.0 * ld_180;
               ld_124 = 0;
               ld_132 = 0;
               ld_140 = 0;
               for (int li_212 = ld_84; li_212 <= ld_108; li_212++) {
                  if (ld_132 == 0.0 && ld_140 >= 3.0 && li_212 > ld_84) {
                     ld_164 = 0;
                     ld_172 = ObjectGet("downtrendline" + li_204 + "tt", OBJPROP_PRICE2);
                     for (int li_216 = 1; li_216 <= 5; li_216++) {
                        if (ld_164 >= 3.0) ld_124 = 1;
                        if (ld_124 == 0.0) {
                           ObjectSet("downtrendline" + li_204 + "tt", OBJPROP_PRICE2, ld_172 + (li_216 - 3) * Point);
                           ld_164 = 0;
                           for (int li_220 = ld_84; li_220 <= ld_108; li_220++) {
                              ld_76 = ObjectGetValueByShift("downtrendline" + li_204 + "tt", li_220);
                              if (ld_76 + ld_180 * Point > iHigh(NULL, 0, li_220) && ld_76 - ld_180 * Point < iHigh(NULL, 0, li_220)) {
                                 ld_164++;
                                 li_220++;
                              }
                           }
                        }
                     }
                  }
                  if (ld_124 == 0.0 && li_212 == ld_108) ObjectDelete("downtrendline" + li_204 + "tt");
                  if (ld_124 == 1.0 && li_212 == ld_108) {
                     ld_148 = ObjectGetValueByShift("downtrendline" + li_204, ld_84);
                     ld_156 = ObjectGetValueByShift("downtrendline" + li_204 + "tt", ld_84);
                     if (MathAbs(ld_148 - ld_156) > ld_188 * Point) ObjectDelete("downtrendline" + li_204 + "tt");
                  }
                  if (ld_124 == 0.0 && li_212 <= ld_108) ObjectMove("downtrendline" + li_204 + "tt", 1, iTime(NULL, 0, li_212), iHigh(NULL, 0, li_212));
                  if (ld_124 == 0.0) {
                     ld_132 = 0;
                     ld_140 = 0;
                     for (li_200 = ld_84; li_200 <= ld_108; li_200++) {
                        ld_76 = ObjectGetValueByShift("downtrendline" + li_204 + "tt", li_200);
                        if (iClose(NULL, 0, li_200) > ObjectGetValueByShift("downtrendline" + li_204 + "tt", li_200)) ld_132++;
                        if (ld_76 + 2.0 * ld_180 * Point > iHigh(NULL, 0, li_200) && ld_76 - 2.0 * ld_180 * Point < iHigh(NULL, 0, li_200)) {
                           ld_140++;
                           li_200++;
                        }
                     }
                  }
               }
            }
         }
         for (li_204 = 1; li_204 <= 30; li_204++) {
            ld_100 = ObjectGet("uptrendline" + li_204, OBJPROP_TIME1);
            ld_108 = iBarShift(NULL, 0, ld_100);
            ld_84 = ld_68;
            ld_116 = ld_108 - ld_84;
            if (ld_116 < MathMin(Normal_TL_Period, 1000) && ld_116 > 6.0) {
               ObjectCreate("uptrendline" + li_204 + "tt", OBJ_TREND, 0, iTime(NULL, 0, ld_108), iLow(NULL, 0, ld_108), iTime(NULL, 0, ld_108), iLow(NULL, 0, ld_108));
               ObjectSet("uptrendline" + li_204 + "tt", OBJPROP_WIDTH, 2);
               ld_180 = iATR(NULL, 0, ld_116, li_196) / Point / 10.0;
               ld_188 = 8.0 * ld_180;
               ld_124 = 0;
               ld_140 = 0;
               for (li_212 = ld_84; li_212 <= ld_108; li_212++) {
                  if (ld_132 == 0.0 && ld_140 >= 3.0 && li_212 > ld_84 && ld_124 == 0.0) {
                     ld_164 = 0;
                     ld_172 = ObjectGet("uptrendline" + li_204 + "tt", OBJPROP_PRICE2);
                     for (li_216 = 1; li_216 <= 5; li_216++) {
                        if (ld_164 >= 3.0) ld_124 = 1;
                        if (ld_124 == 0.0) {
                           ObjectSet("uptrendline" + li_204 + "tt", OBJPROP_PRICE2, ld_172 + (li_216 - 3) * Point);
                           ld_164 = 0;
                           for (li_220 = ld_84; li_220 <= ld_108; li_220++) {
                              ld_76 = ObjectGetValueByShift("uptrendline" + li_204 + "tt", li_220);
                              if (ld_76 + ld_180 * Point > iLow(NULL, 0, li_220) && ld_76 - ld_180 * Point < iLow(NULL, 0, li_220)) {
                                 ld_164++;
                                 li_220++;
                              }
                           }
                        }
                     }
                  }
                  if (ld_124 == 0.0 && li_212 == ld_108) ObjectDelete("uptrendline" + li_204 + "tt");
                  if (ld_124 == 1.0 && li_212 == ld_108) {
                     ld_148 = ObjectGetValueByShift("uptrendline" + li_204, ld_84);
                     ld_156 = ObjectGetValueByShift("uptrendline" + li_204 + "tt", ld_84);
                     if (MathAbs(ld_148 - ld_156) > ld_188 * Point) ObjectDelete("uptrendline" + li_204 + "tt");
                  }
                  if (ld_124 == 0.0 && li_212 < ld_108) ObjectMove("uptrendline" + li_204 + "tt", 1, iTime(NULL, 0, li_212), iLow(NULL, 0, li_212));
                  if (ld_124 == 0.0) {
                     ld_132 = 0;
                     ld_140 = 0;
                     for (li_200 = ld_84; li_200 <= ld_108; li_200++) {
                        ld_76 = ObjectGetValueByShift("uptrendline" + li_204 + "tt", li_200);
                        if (iClose(NULL, 0, li_200) < ObjectGetValueByShift("uptrendline" + li_204 + "tt", li_200)) ld_132++;
                        if (ld_76 + 2.0 * ld_180 * Point > iLow(NULL, 0, li_200) && ld_76 - 2.0 * ld_180 * Point < iLow(NULL, 0, li_200)) {
                           ld_140++;
                           li_200++;
                        }
                     }
                  }
               }
            }
         }
         for (li_200 = 0; li_200 <= 30; li_200++) {
            if (ObjectGetValueByShift("uptrendline" + li_200 + "tt", li_196 + 1) > 0.0) {
               ObjectSet("uptrendline" + li_200, OBJPROP_WIDTH, Three_Touch_TL_Widht);
               ObjectSet("uptrendline" + li_200, OBJPROP_COLOR, Three_Touch_TL_Color);
               ObjectSetText("uptrendline" + li_200, "3t");
               ObjectDelete("uptrendline" + li_200 + "tt");
            }
         }
         for (li_200 = 0; li_200 <= 30; li_200++) {
            if (ObjectGetValueByShift("downtrendline" + li_200 + "tt", li_196 + 1) > 0.0) {
               ObjectSet("downtrendline" + li_200, OBJPROP_WIDTH, Three_Touch_TL_Widht);
               ObjectSet("downtrendline" + li_200, OBJPROP_COLOR, Three_Touch_TL_Color);
               ObjectSetText("downtrendline" + li_200, "3t");
               ObjectDelete("downtrendline" + li_200 + "tt");
            }
         }
      }
      for (li_204 = 0; li_204 <= 30; li_204++) {
         if (ObjectGet("downtrendline" + ((li_204 - 1)), OBJPROP_PRICE1) == 0.0 && ObjectGet("downtrendline" + li_204, OBJPROP_PRICE1) > 0.0 && Mark_Highest_and_Lowest_TL == TRUE) {
            ObjectSet("downmax", OBJPROP_TIME1, iTime(NULL, 0, li_196 + 6));
            ObjectSet("downmax", OBJPROP_PRICE1, ObjectGetValueByShift("downtrendline" + li_204, li_196 + 6));
            ObjectSet("downmax", OBJPROP_TIME2, iTime(NULL, 0, li_196 + 3));
            ObjectSet("downmax", OBJPROP_PRICE2, ObjectGetValueByShift("downtrendline" + li_204, li_196 + 3));
            ObjectSet("downmax", OBJPROP_COLOR, ObjectGet("downtrendline" + li_204, OBJPROP_COLOR));
            ObjectSet("downmax", OBJPROP_WIDTH, 5);
            ObjectSet("downmax", OBJPROP_STYLE, STYLE_SOLID);
            ObjectSet("downmax", OBJPROP_RAY, FALSE);
            ObjectSet("downmax", OBJPROP_BACK, TRUE);
         }
         if (ObjectGet("uptrendline" + ((li_204 - 1)), OBJPROP_PRICE1) == 0.0 && ObjectGet("uptrendline" + li_204, OBJPROP_PRICE1) > 0.0 && Mark_Highest_and_Lowest_TL == TRUE) {
            ObjectSet("upmax", OBJPROP_TIME1, iTime(NULL, 0, li_196 + 6));
            ObjectSet("upmax", OBJPROP_PRICE1, ObjectGetValueByShift("uptrendline" + li_204, li_196 + 6));
            ObjectSet("upmax", OBJPROP_TIME2, iTime(NULL, 0, li_196 + 3));
            ObjectSet("upmax", OBJPROP_PRICE2, ObjectGetValueByShift("uptrendline" + li_204, li_196 + 3));
            ObjectSet("upmax", OBJPROP_COLOR, ObjectGet("uptrendline" + li_204, OBJPROP_COLOR));
            ObjectSet("upmax", OBJPROP_WIDTH, 5);
            ObjectSet("upmax", OBJPROP_STYLE, STYLE_SOLID);
            ObjectSet("upmax", OBJPROP_RAY, FALSE);
            ObjectSet("upmax", OBJPROP_BACK, TRUE);
         }
      }
      ld_232 = 0;
      ld_240 = 0;
      for (li_204 = 1; li_204 <= 30; li_204++) {
         ld_232 += ObjectGet("downtrendline" + li_204, OBJPROP_PRICE1);
         ld_240 += ObjectGet("uptrendline" + li_204, OBJPROP_PRICE1);
      }
      if (ld_232 == 0.0) {
         ObjectSet("downmax", OBJPROP_TIME1, 0);
         ObjectSet("downmax", OBJPROP_PRICE1, 0);
         ObjectSet("downmax", OBJPROP_TIME2, 0);
         ObjectSet("downmax", OBJPROP_PRICE2, 0);
      }
      if (ld_240 == 0.0) {
         ObjectSet("upmax", OBJPROP_TIME1, 0);
         ObjectSet("upmax", OBJPROP_PRICE1, 0);
         ObjectSet("upmax", OBJPROP_TIME2, 0);
         ObjectSet("upmax", OBJPROP_PRICE2, 0);
      }
      ObjectSet("calctl", OBJPROP_PRICE1, 0);
   }
   if (Auto_Refresh == TRUE && IndicatorCounted() == 0) {
      ObjectSet("calctl", OBJPROP_PRICE1, -1);
      li_248 = WindowHandle(Symbol(), Period());
      li_252 = RegisterWindowMessageA("MetaTrader4_Internal_Message");
      PostMessageA(li_248, li_252, 2, 1);
   }
   
    if (rates_total < 2) return (0);

    // Calculate Volume Profile if using volume filter
    if(UseVolumeFilter) {
        CalculateVolumeProfile();
    }

    int limit;
    if (prev_calculated == 0) {
        limit = rates_total - 2;
        ArrayInitialize(BuyBuffer, EMPTY_VALUE);
        ArrayInitialize(SellBuffer, EMPTY_VALUE);
    } else {
        limit = rates_total - prev_calculated + 1;
    }

    static datetime lastUpdateTime = 0;

    for (int i = limit; i >= 0 && !IsStopped(); i--) {
        BuyBuffer[i] = EMPTY_VALUE;
        SellBuffer[i] = EMPTY_VALUE;

        for (int line = 1; line <= 30; line++) {
            // Verifica linhas de alta
            string upName = StringConcatenate("uptrendline", IntegerToString(line));
            if (IsValidTrendLine(upName)) {
                if (IsTrendLineTouch(upName, i, true) && 
                    IsDirectionConfirmed(i, true) &&
                    CheckTouchDistance(upName, i) &&
                    CheckTimeDistance(Time[i]) &&
                    HasMomentum(i, true) &&
                    !IsTrendLineBroken(upName, i, true) &&
                    IsVolumeProfileConfirming(i, true))
                {
                    BuyBuffer[i] = Low[i] - (ArrowDistance * Point);
                    if (Time[i] != lastUpdateTime) {
                        if(EnableAlerts) SendAlerts("COMPRA");
                        lastSignalTime = Time[i];
                        lastUpdateTime = Time[i];
                    }
                }
            }

            // Verifica linhas de baixa
            string downName = StringConcatenate("downtrendline", IntegerToString(line));
            if (IsValidTrendLine(downName)) {
                if (IsTrendLineTouch(downName, i, false) && 
                    IsDirectionConfirmed(i, false) &&
                    CheckTouchDistance(downName, i) &&
                    CheckTimeDistance(Time[i]) &&
                    HasMomentum(i, false) &&
                    !IsTrendLineBroken(downName, i, false) &&
                    IsVolumeProfileConfirming(i, false))
                {
                    SellBuffer[i] = High[i] + (ArrowDistance * Point);
                    if (Time[i] != lastUpdateTime) {
                        if(EnableAlerts) SendAlerts("VENDA");
                        lastSignalTime = Time[i];
                        lastUpdateTime = Time[i];
                    }
                }
            }
        }
    }

    ChartRedraw();
    return (rates_total);
}