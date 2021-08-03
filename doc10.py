import numpy as np
import pandas as pd
import calendar as cd
import datetime as dt
import pdb
import sys 
 
class Doc10SchedulingProblem:
    """This class encapsulates the Nurse Schedulizong problem
    """
 
    def __init__(self, hardConstraintPenalty,df_docs,month,year):
        """
        :param hardConstraintPenalty: the penalty factor for a hard-constraint violation
        """
        self.hardConstraintPenalty = hardConstraintPenalty
 
        # list of doc:
        
        self.docs = list(df_docs['FAM']+' '+ df_docs['NAME'])
        self.df = df_docs
        self.df_dej = self.getRealDejs()
        self.docs_dej = list(self.df_dej['FAM']+' '+ self.df_dej['NAME'])

        # useful values:
        self.days_in_month = cd.monthrange(year,month)[1]
 #       pdb.set_trace()
        self.corps = 3
        self.month = month
        self.year = year
        self.realDejs = self.getRealDejs()
        self.shiftsPerMonth = self.corps*self.days_in_month

    def getDfDocs(self):
        return self.df

    def getDocs(self):
        return self.docs

    def getCorps(self):
        return self.corps

    def getRealDejs(self):
        d = self.df.query('CORPUS!=2 and CORPUS!=0')
        return d
    def getNmbRealDejs(self):
        d = self.getRealDejs()
 #       pdb.set_trace()
        return d.shape[0]

    def getDaysInMonth(self):
        return self.days_in_month
    def getMonth(self):
        return self.month

    def getYear(self):
        return self.year



 
    def __len__(self):
        """
        :return: the number of shifts in the schedule

        """
        return self.corps * self.days_in_month

    def getInitSchedule(self):
        """
        """
        schedule = np.full(len(self),None)
        nmb_dejs = self.getNmbRealDejs()
        sched_days = np.arange(1,self.corps*self.days_in_month+1)
        sched_dejs = np.arange(0,nmb_dejs )
        max_nmb_dej = len(schedule)//nmb_dejs + 1 
        min_dist = 3
        
        cnt=0
#        pdb.set_trace()
        while len(sched_days)>0: 
            
            d = np.random.choice(sched_days)
            dej = np.random.choice(sched_dejs)
#            print(cnt)
 #           print(f'День {self.getDay(d)} корпус {self.getCorpus(d)} дежурант {dej} {self.docs_dej[dej]} стоят {self.getDaysOfDoc(schedule,dej)}')

            cnt+=1

            if self.isSuitableDej(schedule,dej,d, 
                    max_nmb_dej,min_dist):
 #               pdb.set_trace()

                schedule[d-1] = dej

                sched_days = np.delete(sched_days,
                        np.where(sched_days==d))
 #               if self.getDaysOfDoc(schedule,dej)[1] >= max_nmb_dej:
 #                   sched_dejs = np.delete(sched_dejs,
      #                      np.where(sched_dejs == dej))
                
 #               self.printScheduleHuman(schedule)
#                print(f'Присвоили День {self.getDay(d)} корпус {self.getCorpus(d)} дежурант {dej} {self.docs_dej[dej]} стоят {self.getDaysOfDoc(schedule,dej)}')
            

#            input() 
#            pdb.set_trace()
            if cnt >2000:
                print(f'День {self.getDay(d)} корпус {self.getCorpus(d)} дежурант {dej} {self.docs_dej[dej]} стоят {self.getDaysOfDoc(schedule,dej)}')
                print(sched_days)
                self.printScheduleHuman(schedule)
                self.getDejsForDocs(schedule)   
                break 
  #              input()  
#               break
#        print(schedule)

        self.printScheduleHuman(schedule)
        self.getDejsForDocs(schedule)   
      #  printScheduleHumanSum(schedule)

        return schedule


 
    def getCost(self, schedule):
        """
        Calculates the total cost of the various violations in the given schedule
        ...
        :param schedule: a list of binary values describing the given schedule
        :return: the calculated cost
        """
 
 
    def getDocShifts(self, schedule):
        """
        Converts the entire schedule into a dictionary with a separate schedule for each doc
        :param schedule: a list of binary values describing the given schedule
        :return: a dictionary with each doc as a key and the corresponding shifts as the value
        """
        shiftsPerDoc = self.corps*self.days_in_month

        docShiftsDict = {}
        shiftIndex = 0
 
        for doc in self.docs_dej:
            docShiftsDict[doc] = schedule[shiftIndex:shiftIndex + shiftsPerDoc]
            shiftIndex += shiftsPerDoc
 #       import pdb; pdb.set_trace()
 
        return docShiftsDict

    def isSuitableDej(self,schedule,dej, sched_day,max_nmb_dej,min_dist):
        """
        определяет, можно ли поставить день day в
        расписание shedule
        :shedule - расписание , двоичный массив длиной
          дней_в_месяце*количество_корпусов*количество_дежурантов
        :doc - объект DocSchedulingProblem
        :dej_index индекс дежуранта в массиве дежурантов
        :day индекс дня в одномерном массиве shedule
        :return True , если подходит

        """
        
        if not self.isFreeDay(schedule,sched_day):
#            print(f'День {self.getDay(sched_day)} корпус {self.getCorpus(sched_day)} занят') 
            return False

        if not self.isSuitableCorpus(dej,sched_day):
  #          print(f'Корпус {self.getCorpus(sched_day)} не походит') 
            return False
       
        if not self.isSuitableQuantity(schedule,dej,max_nmb_dej):
  #          print('Количество дежурств не подходит ')
            return False


        if not self.isSuitableSequence(schedule,dej,
                sched_day,min_dist):
 #           print('Последовательность дежурств не походит')
            return False

#        print ('Все походит')

        return True



    def isSuitableSequence(self,schedule,dej,sched_day,min_dist):
        pdb.set_trace()
        ar = np.where(schedule == dej)[0]
        ar = ar + 1
        ar0 = np.array([self.getDay(i) for i in ar])
        day = self.getDay(sched_day)
        ar1 = ar0 - day
        ar1 = abs(ar1)
        if np.any(ar1 <= min_dist):
            return False
        else:
            return True


    def isSuitableQuantity(self,schedule, dej, max_nmb_dej):
        n = (schedule==dej).sum()
        if n >= max_nmb_dej:
            return False
        else:
            return True


    def isSuitableCorpus(self,dej, day):

        dej_corp  = self.df_dej.iloc[dej]['CORPUS']
        possible_corpus = self.getCorpus(day) 
        return self.isCorpRight(dej_corp,possible_corpus) 


    def isFreeDay(self,schedule,sched_day):

        if schedule[sched_day-1] == None : 
            return True 
        else:       
            return False



    def printScheduleHuman(self,schedule):
        """
        Выводит расписание в нужном конечном формате
        :schedule ДНК расписания, двоичный 1d массив
        :doc - объект DocSchedulingProblem
        :return требуемая таблица
        """

        corps = self.getCorps()
        days = self.getDaysInMonth()
        fam = self.getRealDejs()['FAM']

        # формируем столбик из дежурантов,
        # попутно добавляем к фамилии И О
        i = self.getRealDejs()['NAME']
        o = self.getRealDejs()['SURNAME']
        dejs = np.array( fam+' '+ i.str[0] +'.' +o.str[0] +'.')
        sch_s = np.full_like(schedule,np.nan)

        cnt = 0

        for x,y in zip(schedule,sch_s):
            if pd.isnull(x) == False:
                sch_s[cnt] = dejs[x]
            cnt += 1

        schedule_with_name = sch_s.reshape(corps,days)
        schedule_with_name = schedule_with_name.transpose()


        # формируем массив дат для конечной таблицы
        dates = np.arange(np.datetime64(dt.date(self.year,
            self.month,1)),days)
        # преобразуем его в столбец
        dates = dates.reshape(-1,1)


        # цепляем столбец с датами слева, попутно преобразуя
        # его в строку
#        pdb.set_trace()
        schedule_with_date = np.hstack(((dates.astype('str')),
           schedule_with_name))

        # nditer позволяет итерировать двумерный массив
        # переносим данные в конечную таблицу,
        # попутно заполняя текстовые поля пробеллами
        # справа до 12

 #       pdb.set_trace()

        df = pd.DataFrame(schedule_with_date,
                columns=['Дата','2_корпус','1_корпус','Хоспис'])   
        print("Расписание дежурств УЗ 'МООД'")
        print("Месяц ",self.getMonth())
        print(df.to_string(index=False))
#        pdb.set_trace()
 #       free_shifts = np.sum(pd.isnull(schedule_with_date))
 #       print('Свободных смен - ',free_shifts)

        return schedule_with_date


    def printScheduleHumanSum(self,schedule):
        """
        Выводит расписание в нужном конечном формате
        :schedule ДНК расписания, двоичный 1d массив
        :doc - объект DocSchedulingProblem
        :return требуемая таблица
        """

        shiftsPerDoc = doc.getCorps()*doc.getDaysInMonth()
        docs_dej = doc.docs_dej

        docShiftsDict = {}
        days = doc.getDaysInMonth()
        shiftIndex = 0
        sched_sum = 0

        for d in docs_dej:
            ar =  np.where(schedule[shiftIndex:shiftIndex 
                + shiftsPerDoc]==1)[0]
            ar = np.fromiter((convFlattenToDejDayCorp(doc,
                i,days)[1] for i in ar),dtype=int)
            sch_s = np.sum(
                    schedule[shiftIndex:shiftIndex + shiftsPerDoc])
        
            docShiftsDict[d] = ar, sch_s
            shiftIndex += shiftsPerDoc
            sched_sum += sch_s
            print()

        for d in docShiftsDict: 
            print(f'{d : <22}  {docShiftsDict[d][1] : ^5}'
                  f'{docShiftsDict[d][0] }')
        print(f'Занято {sched_sum} смен')


    def isCorpRight(self,dej_corp,possible_corpus):
        if possible_corpus not in (1,2,3):
            print('Неправильный номер корпуса -',possible_corpus)
            return False
        if dej_corp==1 and possible_corpus==3:
            return True
        if dej_corp==10 and possible_corpus==2:
            return True
        if dej_corp==100 and possible_corpus==1:
            return True
        if dej_corp==11 and possible_corpus in (2,3):
            return True
        if dej_corp==101 and possible_corpus in (1,3):
            return True
        if dej_corp==110 and possible_corpus in (1,2):
            return True
        if dej_corp==111 and possible_corpus in (1,2,3):
            return True

        return False   

    def getCorpus(self,n):
        days = self.days_in_month
        if n <= 0 or n > days*self.corps:
            print("Неправильный номер дня")
            return
        if n <= days:
            return 1
        if n > days*2:
            return 3
        else:
            return 2

    def getDay(self,n):
        days = self.days_in_month
        if n <= 0 or n > days*self.corps:
            print("Неправильный номер дня")
            return
        if n <= days:
            return n
        if n%days == 0:
            return days
        else:
            return n%days

    def getDaysOfDoc(self,schedule,dej):
        ar = np.where(schedule==dej)[0]
        ret = [ (self.getDay(i+1),self.getCorpus(i+1)) for i in ar ]
        
        return ret,len(ret)

    def getDejsForDocs(self,schedule):
        for i, dej in enumerate(self.docs_dej):
            print(f'{dej: <22} {self.getDaysOfDoc(schedule,i)[1]:^5} '
                  f'{self.getDaysOfDoc(schedule,i)[0]}')





# testing the class:
def main():
    # create a problem instance:
    p = pd.read_csv("lk_1.csv")

    doc  = Doc10SchedulingProblem(10,p,dt.datetime.now().month+1,dt.datetime.now().year)
#    pdb.set_trace()
#    print(dir(doc))
 
    randomSolution = doc.getInitSchedule()

    print("Random Solution = ")
    print(randomSolution)
    print()
    print("Len randomSolution = ", len(randomSolution))
 
   # doc.printScheduleInfo(randomSolution)
 
    print("Total Cost = ", doc.getCost(randomSolution))

 
if __name__ == "__main__":
    main()
 
 
