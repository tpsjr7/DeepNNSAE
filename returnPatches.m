function Patches=returnPatches(X,Y,IMAGE)
Patches=zeros(16,48,1);
TempPatch=zeros(4,4,3);
CNT=0;
for R=X:X+3
    for C=Y:Y+3
        CNT=CNT+1;
        if X<=29
            XS=X;
        elseif X>29 && X<=32
            XS=X-4;
        else
            error('sth wrong in returepatches.m');
        end
        if Y<=29
            YS=Y;
        elseif Y>29 && Y<=32
            YS=Y-4;
        else   
            disp(Y)
            error('sth wrong in returepatches.m');
        end
%         XS=((X<=29 * X)+((X<=32)*(X>29)*(X-4)));
%         YS=((Y<=29 * Y)+((X<=32)*(Y>29)*(Y-4))); 
%         if (XS == 0 || YS == 0)
%             error('sth wrong in returepatches.m');
%         end
%         disp(size(TempPatch))
%         disp(size(IMAGE))
%         pause
%         disp(XS)    
%         disp(YS)
%         pause;
        TempPatch=IMAGE(XS:XS+3,YS:YS+3,:);
        IMG=IMAGE(XS:XS+3,YS:YS+3,:);

%         size(IMAGE(:,:,1))
%         pause
        Temp1=reshape(IMG(:,:,1),16,1);
        Temp2=reshape(IMG(:,:,2),16,1);
        Temp3=reshape(IMG(:,:,3),16,1);
        TempPatch=[Temp1;Temp2;Temp3];%
%         disp(size(TempPatch))
%         disp(size(Patches))
        Patches(CNT,:,:)=TempPatch;
    end
end




end